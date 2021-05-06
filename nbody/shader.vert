#version 330 core

layout (location = 0) in vec3 aPos;

out vec3 ourColor;

uniform vec3 translate;
uniform mat4 view;
uniform mat4 projection;
uniform bool isInterop;

void main()
{
   gl_Position = isInterop ? projection * view * vec4(aPos, 1.0) : projection * view * (vec4(translate, 0.0) + vec4(aPos, 1.0));
   ourColor = vec3(1.0f, 0.5f, 0.5f);
}