//
// Created by zcb on 2023/1/5.
//

#include <GLES3/gl32.h>
#include "Mesh.h"

Mesh::Mesh() {
    this->initializeOpenGLFunctions();
}

void Mesh::ProcessMesh(aiMesh *aiMesh, aiMaterial *aiMaterial) {
    m_pAiMesh = aiMesh;
    m_pAiMaterial = aiMaterial;

    ProcessVertices();
    ProcessFacesIndex();
    ProcessMaterials();

    SetupGLMesh();
}

void Mesh::Render(QOpenGLShaderProgram &shader) {
    QMatrix4x4 translate_matrix = QMatrix4x4();
    // TODO 写成函数
    translate_matrix(0, 0) = m_globalTransform.m[0][0];
    translate_matrix(0, 1) = m_globalTransform.m[1][0];
    translate_matrix(0, 2) = m_globalTransform.m[2][0];
    translate_matrix(0, 3) = m_globalTransform.m[3][0];
    translate_matrix(1, 0) = m_globalTransform.m[0][1];
    translate_matrix(1, 1) = m_globalTransform.m[1][1];
    translate_matrix(1, 2) = m_globalTransform.m[2][1];
    translate_matrix(1, 3) = m_globalTransform.m[3][1];
    translate_matrix(2, 0) = m_globalTransform.m[0][2];
    translate_matrix(2, 1) = m_globalTransform.m[1][2];
    translate_matrix(2, 2) = m_globalTransform.m[2][2];
    translate_matrix(2, 3) = m_globalTransform.m[3][2];
    translate_matrix(3, 0) = m_globalTransform.m[0][3];
    translate_matrix(3, 1) = m_globalTransform.m[1][3];
    translate_matrix(3, 2) = m_globalTransform.m[2][3];
    translate_matrix(3, 3) = m_globalTransform.m[3][3];

    shader.setUniformValue("translate", translate_matrix);
    shader.setUniformValue("material.ambient", m_materials.ambient);
    shader.setUniformValue("material.diffuse", m_materials.diffuse);
    shader.setUniformValue("material.specular", m_materials.specular);

    // Render Texture
    if (textures.size() != 0) {
        shader.setUniformValue("isUseTexture", true);
        for (int i = 0; i < textures.size(); ++i) {
            glActiveTexture(GL_TEXTURE0 + i);
            shader.setUniformValue("texture_diffuse", i);
            glBindTexture(GL_TEXTURE_2D, textures[i].id);
        }
    } else {
        shader.setUniformValue("isUseTexture", false);
    }

    glBindVertexArray(m_MeshVAO);
    glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void Mesh::SetupGLMesh() {
    // create buffers/arrays
    glGenVertexArrays(1, &m_MeshVAO);
    glGenBuffers(1, &m_MeshVBO);
    glGenBuffers(1, &m_MeshEBO);

    glBindVertexArray(m_MeshVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_MeshVBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_MeshEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

    GLsizei stride = sizeof(Vertex);
    // set the vertex attribute pointers
    // vertex Positions
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void *) 0);
    // vertex normals
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void *) offsetof(Vertex, Normal));
    // vertex texture coords
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, (void *) offsetof(Vertex, TexCoords));
}

void Mesh::ProcessVertices() {
    for (int i = 0; i < m_pAiMesh->mNumVertices; ++i) {
        Vertex vertex; //创建一个Vertex存储顶点数据
        // Postion
        if (m_pAiMesh->HasPositions()) {
            aiVector3D aiPos = m_pAiMesh->mVertices[i];
            vertex.Position = {aiPos.x, aiPos.y, aiPos.z};
        }

        // Normal
        if (m_pAiMesh->HasNormals()) {
            aiVector3D aiNormal = m_pAiMesh->mNormals[i];
            vertex.Normal = {aiNormal.x, aiNormal.y, aiNormal.z};
        }

        // TextureCoords
        if (m_pAiMesh->HasTextureCoords(0)) {
            aiVector3D aiUv = m_pAiMesh->mTextureCoords[0][i];
            vertex.TexCoords = QVector2D(aiUv.x, aiUv.y);
        } else {
            vertex.TexCoords = QVector2D(0.0f, 0.0f);
        }

        vertices.push_back(vertex);
    }
}

void Mesh::ProcessFacesIndex() {
    for (size_t i = 0; i < m_pAiMesh->mNumFaces; ++i) {
        aiFace aiFace = m_pAiMesh->mFaces[i];
        for (size_t j = 0; j < aiFace.mNumIndices; ++j) {
            indices.push_back(aiFace.mIndices[j]);
        }
    }
}

void Mesh::ProcessMaterials() {
    aiColor3D color;
    m_pAiMaterial->Get(AI_MATKEY_COLOR_AMBIENT, color);
    m_materials.ambient = QVector3D(color.r, color.g, color.b);
    m_pAiMaterial->Get(AI_MATKEY_COLOR_DIFFUSE, color);
    m_materials.diffuse = QVector3D(color.r, color.g, color.b);
    m_pAiMaterial->Get(AI_MATKEY_COLOR_SPECULAR, color);
    m_materials.specular = QVector3D(color.r, color.g, color.b);
}
