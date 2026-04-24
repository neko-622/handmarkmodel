/*
 * @Filename: demo_face.cpp
 * @Author: Hongying He
 * @Email: hongying.he@smartsenstech.com
 * @Date: 2025-12-30 14-57-47
 * @Copyright (c) 2025 SmartSens
 */
#include <fstream>
#include <iostream>
#include <cstring>
#include <thread>
#include <mutex>
#include <fcntl.h>
#include <regex>
#include <dirent.h>
#include <unistd.h>
#include "include/utils.hpp"

using namespace std;

// 全局退出标志（线程安全）
bool g_exit_flag = false;
// 保护退出标志的互斥锁
std::mutex g_mtx;

// OSD 贴图结构体
struct osdInfo {
    std::string filename; // OSD 文件名
    uint16_t x;           // 起始坐标 x
    uint16_t y;           // 起始坐标 y
};

/**
 * @brief 键盘监听程序，用于结束demo
 */
void keyboard_listener() {
    std::string input;
    std::cout << "键盘监听线程已启动，输入 'q' 退出程序..." << std::endl;

    while (true) {
        // 读取键盘输入（会阻塞直到有输入）
        std::cin >> input;

        // 加锁修改退出标志
        std::lock_guard<std::mutex> lock(g_mtx);
        if (input == "q" || input == "Q") {
            g_exit_flag = true;
            std::cout << "检测到退出指令，通知主线程退出..." << std::endl;
            break;
        } else {
            std::cout << "输入无效（仅 'q' 有效），请重新输入：" << std::endl;
        }
    }
}

/**
 * @brief 检查退出标志的辅助函数（线程安全）
 * @return 是否需要退出
 */
bool check_exit_flag() {
    std::lock_guard<std::mutex> lock(g_mtx);
    return g_exit_flag;
}

/**
 * @brief 绘制手部关键点
 * @param visualizer 可视化器实例
 * @param keypoints 手部关键点坐标
 * @param crop_offset_y 裁剪偏移量
 */
void draw_hand_keypoints(VISUALIZER& visualizer, const std::vector<std::array<float, 2>>& keypoints, int crop_offset_y) {
    // 定义关键点连接顺序（根据手部骨骼结构）
    const std::vector<std::pair<int, int>> connections = {
        {0, 1}, {1, 2}, {2, 3}, {3, 4},  // 拇指
        {0, 5}, {5, 6}, {6, 7}, {7, 8},  // 食指
        {0, 9}, {9, 10}, {10, 11}, {11, 12},  // 中指
        {0, 13}, {13, 14}, {14, 15}, {15, 16},  // 无名指
        {0, 17}, {17, 18}, {18, 19}, {19, 20}   // 小指
    };

    // 绘制连接线
    for (const auto& conn : connections) {
        int idx1 = conn.first;
        int idx2 = conn.second;
        if (idx1 < keypoints.size() && idx2 < keypoints.size()) {
            // 转换到原图坐标
            float x1 = keypoints[idx1][0];
            float y1 = keypoints[idx1][1] + crop_offset_y;
            float x2 = keypoints[idx2][0];
            float y2 = keypoints[idx2][1] + crop_offset_y;

            // 绘制线段（使用小矩形模拟线段）
            float width = 2.0f;
            std::array<float, 4> line_box = {std::min(x1, x2) - width, std::min(y1, y2) - width, 
                                           std::max(x1, x2) + width, std::max(y1, y2) + width};
            std::vector<std::array<float, 4>> line_boxes = {line_box};
            visualizer.Draw(line_boxes);
        }
    }

    // 绘制关键点（使用小正方形）
    for (const auto& keypoint : keypoints) {
        float x = keypoint[0];
        float y = keypoint[1] + crop_offset_y;
        float size = 3.0f;
        std::array<float, 4> point_box = {x - size, y - size, x + size, y + size};
        std::vector<std::array<float, 4>> point_boxes = {point_box};
        visualizer.Draw(point_boxes);
    }
}

/**
 * @brief 手部检测演示程序主函数
 * @return 执行结果，0表示成功
 */
int main() {
    /******************************************************************************************
     * 1. 参数配置
     ******************************************************************************************/

    // 图像尺寸配置（根据镜头参数修改）
    int img_width = 720;    // 输入图像宽度
    int img_height = 1280;  // 输入图像高度

    // 模型配置参数
    array<int, 2> det_shape = {224, 224};  // 检测模型输入尺寸
    string path_det = "/app_demo/cut_model.onnx";  // 手部检测模型路径

    // OSD 信息
    static osdInfo osds[3] = {
        {"si.ssbmp", 10, 10},
        {"te.ssbmp", 90, 10},
        {"wei.ssbmp", 170, 10}
    };

    /******************************************************************************************
     * 2. 系统初始化
     ******************************************************************************************/

    // SSNE初始化
    if (ssne_initial()) {
        fprintf(stderr, "SSNE initialization failed!\n");
    }

    // 图像处理器初始化
    array<int, 2> img_shape = {img_width, img_height};  // 原始图像尺寸
    array<int, 2> crop_shape = {224, 224};  // 裁剪尺寸（适应手部模型输入）
    const int crop_offset_y = 528;  // 裁剪时Y方向的偏移量
    // 原图: 720×1280, 模型输入图：224×224
    // 为了保证模型输入图经过resize后比例合适，需要先将原图裁剪为crop图: 224×224

    IMAGEPROCESSOR processor;
    processor.Initialize(&img_shape);  // 初始化图像处理器（配置原图尺寸）

    // 手部检测模型初始化
    SCRFDGRAY detector;
    int box_len = det_shape[0] * det_shape[1] / 512 * 21;  // 计算最大检测框数量
    detector.Initialize(path_det, &crop_shape, &det_shape, false, box_len);  // 初始化检测器

    // 手部检测结果初始化
    FaceDetectionResult* det_result = new FaceDetectionResult;

    // OSD可视化器初始化（用于绘制检测框）
    VISUALIZER visualizer;
    //visualizer.Initialize(img_shape);  // 初始化可视化器（配置图像尺寸）
    visualizer.Initialize(img_shape, "shared_colorLUT.sscl");  // 初始化可视化器（配置图像尺寸和位图LUT）

    // 系统稳定等待
    cout << "sleep for 0.2 second!" << endl;
    sleep(0.2);  // 等待系统稳定

    // OSD 贴图
    visualizer.DrawBitmap(osds[0].filename, "shared_colorLUT.sscl", osds[0].x, osds[0].y, 2);

    uint16_t num_frames = 0;  // 帧计数器
    uint8_t osd_index = 0; // osd 贴图 index
    ssne_tensor_t img_sensor;  // 图像tensor定义

    // 创建键盘监听线程
    std::thread listener_thread(keyboard_listener);

    /******************************************************************************************
     * 3. 主处理循环
     ******************************************************************************************/
    //循环50000帧后推出，循环次数可以修改，也可以改成while(true)
    while (!check_exit_flag()) {

        // 从sensor获取图像（裁剪图）
        processor.GetImage(&img_sensor);

        // 手部检测模型推理（置信度阈值0.4）
        detector.Predict(&img_sensor, det_result, 0.4f);

        /**********************************************************************************
         * 3.1 判断是否有检测到手部
         **********************************************************************************/
        if (det_result->boxes.size() > 0) {
            /**********************************************************************************
             * 3.2 坐标转换：将crop图坐标转换为原图坐标
             **********************************************************************************/
            std::vector<std::array<float, 4>> boxes_original_coord;  // 存储转换后的原图坐标
            std::vector<std::array<float, 2>> keypoints;  // 存储手部关键点坐标

            // 遍历所有检测框进行坐标转换
            for (size_t i = 0; i < det_result->boxes.size(); i++) {
                // 原始crop图坐标（基于224×224裁剪图）
                float x1_crop = det_result->boxes[i][0];  // 左上角x
                float y1_crop = det_result->boxes[i][1];  // 左上角y
                float x2_crop = det_result->boxes[i][2];  // 右下角x
                float y2_crop = det_result->boxes[i][3];  // 右下角y

                // 转换到原图坐标（y坐标加上裁剪偏移，x坐标不变）
                float x1_orig = x1_crop;
                float y1_orig = y1_crop + crop_offset_y;  // 加上裁剪偏移量528
                float x2_orig = x2_crop;
                float y2_orig = y2_crop + crop_offset_y;

                // 保存原图坐标用于OSD绘制
                boxes_original_coord.push_back({x1_orig, y1_orig, x2_orig, y2_orig});
            }

            // 提取手部关键点（假设模型输出包含21个关键点）
            // 这里需要根据实际模型输出格式进行调整
            // 假设关键点存储在det_result->landmarks中
            if (det_result->landmarks.size() >= 21) {
                for (int i = 0; i < 21; i++) {
                    if (i < det_result->landmarks.size()) {
                        float x = det_result->landmarks[i][0];
                        float y = det_result->landmarks[i][1];
                        keypoints.push_back({x, y});
                    }
                }
            }

            /**********************************************************************************
             * 3.3 OSD绘图：使用原图坐标在OSD上绘制手部检测框和关键点
             **********************************************************************************/
            visualizer.Draw(boxes_original_coord);
            draw_hand_keypoints(visualizer, keypoints, crop_offset_y);
        }
        else {
            // 未检测到手部，清除OSD上的检测框
            cout << "[INFO] No hand detected" << endl;
            std::vector<std::array<float, 4>> empty_boxes;
            visualizer.Draw(empty_boxes);  // 传入空向量清除显示
        }

        num_frames += 1;  // 帧计数器递增

        // OSD 贴图
        osd_index = (num_frames / 10) % 3;
        visualizer.DrawBitmap(osds[osd_index].filename, "shared_colorLUT.sscl", osds[osd_index].x, osds[osd_index].y, 2);
    }

    // 等待监听线程退出，释放资源
    if (listener_thread.joinable()) {
        listener_thread.join();
    }

    /******************************************************************************************
     * 4. 资源释放
     ******************************************************************************************/

    delete det_result;  // 释放检测结果
    detector.Release();  // 释放检测器资源
    processor.Release();  // 释放图像处理器资源
    visualizer.Release();  // 释放可视化器资源

    if (ssne_release()) {
        fprintf(stderr, "SSNE release failed!\n");
        return -1;
    }

    return 0;
}

