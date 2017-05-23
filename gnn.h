#ifndef GUO_ANN_2017_5_23_12_53_27366_
#define GUO_ANN_2017_5_23_12_53_27366_

#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <dirent.h>
#include <unistd.h>
#include <vector>
#include <sstream>
#include <fstream>
#include <sys/io.h>
#include <sys/times.h>
#include <cmath>
using namespace std;

/********************************************************************
* BP Nerual Networks
* URL: http://blog.csdn.net/ironyoung/article/details/49455343
*/

#define innode          2		// 输入结点数
#define hidenode        4		// 隐含结点数
#define hidelayer       1		// 隐含层数
#define outnode         1		// 输出结点数
#define learningRate    0.9	    // 学习速率，alpha

// --- -1~1 随机数产生器 ---
inline double get_11Random()    // -1 ~ 1
{
    return ((2.0*(double)rand() / RAND_MAX) - 1);
}

// --- sigmoid 函数 ---
inline double sigmoid(double x)
{
    double ans = 1 / (1 + exp(-x));
    return ans;
}

// --- 输入层节点。包含以下分量：---
// 1.value:     固定输入值；
// 2.weight:    面对第一层隐含层每个节点都有权值；
// 3.wDeltaSum: 面对第一层隐含层每个节点权值的delta值累积
typedef struct inputNode
{
    double value;
    std::vector<double> weight, wDeltaSum;
}inputNode;

// --- 输出层节点。包含以下数值：---
// 1.value:     节点当前值；
// 2.delta:     与正确输出值之间的delta值；
// 3.rightout:  正确输出值
// 4.bias:      偏移量
// 5.bDeltaSum: bias的delta值的累积，每个节点一个
typedef struct outputNode   // 输出层节点
{
    double value, delta, rightout, bias, bDeltaSum;
}outputNode;

// --- 隐含层节点。包含以下数值：---
// 1.value:     节点当前值；
// 2.delta:     BP推导出的delta值；
// 3.bias:      偏移量
// 4.bDeltaSum: bias的delta值的累积，每个节点一个
// 5.weight:    面对下一层（隐含层/输出层）每个节点都有权值；
// 6.wDeltaSum： weight的delta值的累积，面对下一层（隐含层/输出层）每个节点各自积累
typedef struct hiddenNode   // 隐含层节点
{
    double value, delta, bias, bDeltaSum;
    std::vector<double> weight, wDeltaSum;
}hiddenNode;

// --- 单个样本 ---
typedef struct sample
{
    std::vector<double> in, out;
}sample;

// --- BP神经网络 ---
class BpNet
{
public:
    BpNet();    //构造函数
    void forwardPropagationEpoc();  // 单个样本前向传播
    void backPropagationEpoc();     // 单个样本后向传播

    void training( std::vector<sample> sampleGroup, double threshold);// 更新 weight, bias
    void predict(std::vector<sample>& testGroup);                          // 神经网络预测

    void setInput( std::vector<double> sampleIn);     // 设置学习样本输入
    void setOutput( std::vector<double> sampleOut);    // 设置学习样本输出

public:
    double error;
    inputNode* inputLayer[innode];                      // 输入层（仅一层）
    outputNode* outputLayer[outnode];                   // 输出层（仅一层）
    hiddenNode* hiddenLayer[hidelayer][hidenode];       // 隐含层（可能有多层）
};



















#endif // GUO_ANN_2017_5_23_12_53_27366_
