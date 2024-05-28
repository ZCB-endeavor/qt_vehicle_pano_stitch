#version 330 core

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoords;
out vec4 FragColor;

struct Materials {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    vec3 emission;
    float shininess;
};

uniform vec3 viewPos;
uniform Materials material;
uniform sampler2D texture_diffuse;
// 控制是否显示贴图
bool switchTexure = true;
uniform bool isUseTexture;

void main() {
    if (isUseTexture && switchTexure) {
        FragColor = vec4(texture(texture_diffuse, TexCoords).rgb, 1.0);
    } else {
        // ambient
        vec3 ambient = material.ambient * vec3(0.3);

        // diffuse
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(vec3(10000, 6000, 3000) - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = vec3(0.8) * (diff * material.diffuse);

        vec3 result = ambient + diffuse;
        FragColor = vec4(result, 1.0);
    }
}