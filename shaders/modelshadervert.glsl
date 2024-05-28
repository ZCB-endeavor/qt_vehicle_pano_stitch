#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoords;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
uniform mat4 translate;

void main()
{
    gl_Position = projection * view * model * translate * vec4(aPos,1.0);
    FragPos = vec3(model * translate * vec4(aPos, 1.0));
    Normal = aNormal;
    TexCoords = aTexCoords;
}