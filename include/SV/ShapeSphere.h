//
// Created by zcb on 2023/5/30.
//

#ifndef OGLTEST_SHAPESPHERE_H
#define OGLTEST_SHAPESPHERE_H

#include <vector>
#include <cmath>
#include <memory>

class ShapeSphere {
public:
    typedef std::shared_ptr<ShapeSphere> Ptr;

    ShapeSphere() {
        init();
    }

    // 球体坐标系 向上为z轴正方向 向右为x轴正方向 向内为y轴正方向
    // Qt坐标系与glfw坐标系不同，Qt坐标系向上为y轴正方向 向右为z轴正方向 向内为x轴正方向
    void init() {
        float PI = 3.1415926;
        // 均分360度的圆形角度, 即每层圆形的所有顶点坐标
        float sectorStep = 2.f * PI / sectorCount;
        // 均分180度的球体角度, 即上半球到下半球
        float stackStep = PI / stackCount;
        float sectorAngle = 0;
        float stackAngle = 0;
        // 所有层圆形构成球体
        for (int i = 0; i <= stackCount; ++i) {
            // 该角度代表投影到xz平面上的直线与x轴正方向的夹角度数, 从球体最上层的一个顶点开始
            stackAngle = PI / 2 - i * stackStep;
            float xz = m_radius * cos(stackAngle);
            // 让球心坐标沿z轴正方向移动一些距离
//            float y = m_radius * sin(stackAngle);
            float cut_area = 5.0f * m_radius / 6.0f;
            float y = m_radius * sin(stackAngle) + cut_area;
            if (y < 0.0f) {
                y = 0.0f;
            }
            // 每层圆形顶点坐标
            for (int j = 0; j <= sectorCount; ++j) {
                // 该角度代表投影到xy平面上的直线与x轴正方向的夹角度数
                sectorAngle = j * sectorStep;
                float z = xz * cos(sectorAngle);
                float x = xz * sin(sectorAngle);
                // 顶点坐标
                vertices.push_back(x);
                vertices.push_back(y);
                vertices.push_back(z);
                // 归一化顶点坐标
                normals.push_back(x / m_radius);
                normals.push_back(y / m_radius);
                normals.push_back(z / m_radius);
                // 纹理坐标, 值域为[0, 1]
                float s = (float) j / sectorCount;
                float t = (float) i / stackCount;
                texCoords.push_back(s);
                texCoords.push_back(t);
            }
        }

        int k1, k2;
        for (int i = 0; i < stackCount; ++i) {
            k1 = i * (sectorCount + 1);     // beginning of current stack
            k2 = k1 + sectorCount + 1;      // beginning of next stack
            for (int j = 0; j < sectorCount; ++j, ++k1, ++k2) {
                // k1 => k2 => k1+1
                indices.push_back(k1);
                indices.push_back(k2);
                indices.push_back(k1 + 1);

                // k1+1 => k2 => k2+1
                indices.push_back(k1 + 1);
                indices.push_back(k2);
                indices.push_back(k2 + 1);
            }
        }
        m_nVertice = vertices.size();
        m_nIndices = indices.size();
    }

public:
    std::vector<float> vertices;
    std::vector<float> normals;
    std::vector<float> texCoords;
    std::vector<unsigned int> indices;
    int m_nVertice;
    int m_nIndices;
    float m_radius = 6.f;
    int sectorCount = 64;
    int stackCount = 64;
    unsigned int VAO;
    unsigned int VBO[3];
    unsigned int EBO;
};

#endif //OGLTEST_SHAPESPHERE_H
