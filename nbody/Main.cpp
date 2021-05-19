#include "Methods.h"

float lastX = WIDTH / 2.0;
float lastY = HEIGHT / 2.0;
bool firstMouse = true;
Camera camera = Camera(glm::vec3(0.0f, 0.0f, 3.0f));

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	camera.ProcessMouseScroll(yoffset);
}

void mouse_callback(GLFWwindow* window, double xPos, double yPos)
{
	if (firstMouse)
	{
		lastX = xPos;
		lastY = yPos;
		firstMouse = false;
	}

	float xoffset = xPos - lastX;
	float yoffset = lastY - yPos;
	lastX = xPos;
	lastY = yPos;
	camera.ProcessMouseMovement(xoffset, yoffset);
}

/*
Input
	0: CPU mode
	1: CUDA mode
	2: INTEROP mode
	3: BINNING mode
*/
int main(int argc, char **argv)
{
	if (argc < 2)
	{
		std::cout << "0: CPU mode(default), 1 : CUDA mode, 2 : INTEROP mode" << std::endl;
	}

	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

	GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, TITLE, NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetScrollCallback(window, scroll_callback);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	if (argc > 1)
	{
		if (strncmp(argv[1], "0", 1) == 0)
		{
			std::cout << "CPU MODE" << std::endl;
			return cpu_mode(window);
		}
		else if (strncmp(argv[1], "1", 1) == 0)
		{
			std::cout << "CUDA MODE" << std::endl;
			return cuda_mode(window);
		}
		else if (strncmp(argv[1], "2", 1) == 0)
		{
			std::cout << "INTEROP MODE" << std::endl;
			return interop_mode(window);
		}
		else if (strncmp(argv[1], "3", 1) == 0)
		{
			std::cout << "BINNING MODE" << std::endl;
			return bin_mode(window);
		}
		else
		{
			std::cout << argv[1] << std::endl;
			return EXIT_FAILURE;
		}
	}
	
	return 0;
}