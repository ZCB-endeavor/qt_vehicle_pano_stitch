//
// Created by zcb on 2023/1/5.
//

#include <QFile>
#include "Model.h"

Model::Model(const QString &modelPath) {
    qDebug() << "Load Model: " << modelPath;
    m_directory = modelPath.mid(0, modelPath.lastIndexOf('/'));
    this->LoadModel(modelPath);
}

bool Model::LoadModel(const QString &modelPath) {
    m_pScene = importer.ReadFile(modelPath.toStdString(), aiProcess_Triangulate);

    if (!m_pScene || !m_pScene->mRootNode || m_pScene->mFlags == AI_SCENE_FLAGS_INCOMPLETE) {
        qDebug() << "Can not load model file: " << modelPath;
        qDebug() << "ERROR::ASSIMP::" << importer.GetErrorString();
        return false;
    }

    ProcessScene();
    return true;
}

void Model::ProcessScene() {
    // 保存root的translate
    m_rootNodeTransform = m_pScene->mRootNode->mTransformation;
    m_rootNodeTransformInv = m_rootNodeTransform.Inverse();
    // 递归方式处理aiScene里面的所有数据
    ProcessNode(m_pScene->mRootNode, m_rootNodeTransform);
}

void Model::ProcessNode(aiNode *node, const Matrix4f parentTransform) {
    Matrix4f NodeTransformation(node->mTransformation);
    Matrix4f currGlobalTransfrom = parentTransform * NodeTransformation;

    for (int i = 0; i < node->mNumMeshes; ++i) {
        //! node里面的mMeshes只是索引，真正数据在aiScene里面的mMeshes
        aiMesh *aiMesh = m_pScene->mMeshes[node->mMeshes[i]];
        aiMaterial *aiMaterial = m_pScene->mMaterials[aiMesh->mMaterialIndex];
        Mesh mesh;
        mesh.m_globalTransform = currGlobalTransfrom;
        mesh.ProcessMesh(aiMesh, aiMaterial);
        ProcessTexture(aiMaterial, mesh.textures);
        m_Meshes.push_back(mesh);
    }

    // 处理孩子节点
    for (size_t i = 0; i < node->mNumChildren; ++i) {
        ProcessNode(node->mChildren[i], currGlobalTransfrom);
    }
}

void Model::ProcessTexture(aiMaterial *aiMaterial, QVector<Texture> &textures) {
    // 处理满反射diffuse贴图
    QVector<Texture> diffuseMaps = LoadTexture(aiMaterial, aiTextureType_DIFFUSE);
    for (const auto &t: diffuseMaps) {
        textures.push_back(t);
    }
}

QVector<Texture> Model::LoadTexture(aiMaterial *pMaterial, aiTextureType type) {
    QVector<Texture> textures;

    for (GLuint i = 0; i < pMaterial->GetTextureCount(type); i++) {
        aiString str;
        pMaterial->GetTexture(type, i, &str);

        GLboolean skip = false;
        for (GLuint j = 0; j < textures_loaded.size(); j++) {
            if (std::strcmp(textures_loaded[j].path.toStdString().data(), str.C_Str()) == 0) {
                textures.push_back(textures_loaded[j]);
                skip = true;
                break;
            }
        }
        if (!skip) {
            Texture texture;
            QString texturePath = str.C_Str();
            bool isHaveTexture = TextureFromFile(texturePath, texture.id);
            if (isHaveTexture) {
                texture.path = str.C_Str();
                textures.push_back(texture);
                textures_loaded.push_back(texture);
            }
        }
    }
    return textures;
}

bool Model::TextureFromFile(QString &path, unsigned int &textureID) {
    qDebug() << "TextureFromFile path: " << path;
    QString tailPath1 = path.mid(path.lastIndexOf('/') + 1, path.size());
    QString tailPath2 = path.mid(path.lastIndexOf('\\') + 1, path.size());
    QString relativePath = tailPath1.size() > tailPath2.size() ? tailPath2 : tailPath1;
    QString filename = m_directory + '/' + relativePath;

    int width, height, channels;
    stbi_set_flip_vertically_on_load(true);
    unsigned char *data = stbi_load(filename.toStdString().c_str(), &width, &height, &channels, 0);

    if (data) {
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        GLenum format;
        switch (channels) {
            case 1:
                format = GL_RED;
                break;
            case 3:
                format = GL_RGB;
                break;
            case 4:
                format = GL_RGBA;
                break;
            default:
                break;
        }
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, 0);
        qDebug() << "Texture Success: " << filename << " == ID: " << textureID << " == ";
        stbi_image_free(data);
        return true;
    } else {
        qDebug() << "Texture failed: " << filename << " == ID: " << textureID << " == ";
        stbi_image_free(data);
        return false;
    }
}

void Model::Render(QOpenGLShaderProgram &shader) {
    for (size_t i = 0; i < m_Meshes.size(); ++i) {
        m_Meshes[i].Render(shader);
    }
}
