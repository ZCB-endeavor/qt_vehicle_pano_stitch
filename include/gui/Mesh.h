//
// Created by zcb on 2023/1/5.
//

#ifndef TESTQT_MESH_H
#define TESTQT_MESH_H

#include <QVector3D>
#include <QString>
#include <QVector>
#include <QOpenGLFunctions>
#include <QOpenGLBuffer>
#include <QOpenGLShader>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "ogldev_util.h"
#include "ogldev_math_3d.h"

struct Vertex {
    // position
    QVector3D Position;
    // normal
    QVector3D Normal;
    // texCoords
    QVector2D TexCoords;
};

struct Materials {
    QVector3D ambient;
    QVector3D diffuse;
    QVector3D specular;
    QVector3D emission;
    float shininess;
};

struct Texture {
    unsigned int id;
    QString path;
};

class Mesh : protected QOpenGLFunctions {
public:
    Mesh();

    void ProcessMesh(aiMesh *aiMesh, aiMaterial *aiMaterial);

    void Render(QOpenGLShaderProgram &shader);

    Materials m_materials;
    QVector<Texture> textures;
    Matrix4f m_globalTransform;
    unsigned int m_MeshVAO;
private:
    void ProcessVertices();

    void ProcessFacesIndex();

    void ProcessMaterials();

    void SetupGLMesh();

    aiMesh *m_pAiMesh;
    aiMaterial *m_pAiMaterial;
    QVector<Vertex> vertices;
    QVector<GLuint> indices;
    unsigned int m_MeshVBO;
    unsigned int m_MeshEBO;
};

#endif //TESTQT_MESH_H
