//
// Created by zcb on 2023/1/5.
//

#ifndef TESTQT_MODEL_H
#define TESTQT_MODEL_H

#include <QMap>
#include <QOpenGLTexture>
#include <QOpenGLFunctions>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "Mesh.h"
#include "stb_image.h"

class Model : protected QOpenGLFunctions {
public:
    Model(const QString &path);

    bool LoadModel(const QString &modelPath);

    void ProcessScene();

    void ProcessNode(aiNode *node, Matrix4f parentTransform);

    void Render(QOpenGLShaderProgram &shader);

    QVector<Mesh> m_Meshes;
private:
    bool TextureFromFile(QString &path, unsigned int &textureID);

    QVector<Texture> LoadTexture(aiMaterial *pMaterial, aiTextureType type);

    void ProcessTexture(aiMaterial *aiMaterial, QVector<Texture> &textures);

    const aiScene *m_pScene;
    Assimp::Importer importer;
    Matrix4f m_rootNodeTransform;
    Matrix4f m_rootNodeTransformInv;
    QVector<Texture> textures_loaded;
    QString m_directory;
};

#endif //TESTQT_MODEL_H
