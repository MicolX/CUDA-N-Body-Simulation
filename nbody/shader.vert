#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 bPos;

out vec3 ourColor;

uniform vec3 translate;
uniform mat4 view;
uniform mat4 projection;
uniform bool isInterop;
uniform bool isBox;

void main()
{
	if (isBox)
	{
		gl_Position = projection * view * vec4(bPos, 1.0);
		ourColor = vec3(0.0f, 0.7f, 0.93f);
	}
	else
	{
		gl_Position = isInterop ? projection * view * vec4(aPos, 1.0) : projection * view * (vec4(translate, 0.0) + vec4(aPos, 1.0));
		ourColor = vec3(1.0f, 0.5f, 0.5f);
	}
	
}