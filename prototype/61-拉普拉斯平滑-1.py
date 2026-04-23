from laplacian import *
from tqdm import tqdm

verts, faces = resolve_input('../input/kitten.obj')

adj_mat = get_adj_mat(faces)


def forward_euler_smooth_stable(verts, adj_mat, steps=50, lam=0.1):
    v = verts.copy()
    for i in tqdm(range(steps), desc="Forward Euler Smooth"):
        lplc = get_laplacians(v, faces, adj_mat)
        v = v + lam * lplc
    return v


def forward_euler_smooth_origin(verts, adj_mat, steps=50, lam=0.1):
    v = verts.copy()
    laplacians = get_laplacians(v, faces, adj_mat)

    for i in tqdm(range(steps), desc="Forward Euler Smooth"):
        v = v + lam * laplacians

    return v


def backward_euler_smooth(verts, adj_mat, steps=10, lam=0.1):
    v = verts.copy()
    L = get_laplacian_matrix(v, adj_mat)
    n = L.shape[0]
    I = sp.eye(n)
    A = I - lam * L

    for _ in tqdm(range(steps)):
        v_new = np.zeros_like(v)
        for d in range(3):
            v_new[:, d] = sp.linalg.spsolve(A, v[:, d])
        v = v_new

    return v


def backward_euler_smooth_fast(verts, adj_mat, steps=10, lam=0.1):
    v = verts.copy()
    L = get_laplacian_matrix(v, adj_mat)
    n = L.shape[0]
    I = sp.eye(n)
    A = (I - lam * L).tocsc()
    solver = sp.linalg.factorized(A)

    for _ in tqdm(range(steps)):
        for d in range(3):
            v[:, d] = solver(v[:, d])

    return v


# verts = forward_euler_smooth_stable(verts, adj_mat, steps=20)
# verts = forward_euler_smooth_origin(verts, adj_mat, steps=20)
verts = backward_euler_smooth_fast(verts, adj_mat, steps=20)





laplacians = get_laplacians(verts, faces, adj_mat)
scale = np.empty(len(verts), dtype=np.float32)
def get_vertex_normals(verts, faces):
    v_normals = np.zeros_like(verts)
    for f in faces:
        v0, v1, v2 = verts[f]
        # 叉积得到面法线
        face_normal = np.cross(v1 - v0, v2 - v0)
        # 累加到三个顶点上（可以按面积加权，这里简化处理）
        v_normals[f] += face_normal

    # 归一化
    norms = np.linalg.norm(v_normals, axis=1, keepdims=True)
    return v_normals / (norms + 1e-9)


vertex_normals = get_vertex_normals(verts, faces)

# 2. 判断方向
for p in range(len(verts)):
    scale[p] = np.linalg.norm(laplacians[p])
    # 如果拉普拉斯向量与法线方向相反（点积为负），则设为负值
    if np.dot(laplacians[p], vertex_normals[p]) < 0:
        scale[p] *= -1






from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GL.shaders import compileProgram, compileShader

# 准备颜色
scale_abd_max = np.abs(scale).max()
normed_scale = scale / scale_abd_max
def blue_gray_red(t):
    if t < 0:
        # 蓝 -> 灰
        a = -t
        # a **= 0.5
        # (a)(0,0,1) + (1-a)(0.5,0.5,0.5)
        return (0.5-0.5*a, 0.5-0.5*a, 0.5+0.5*a)
    else:
        # 灰 -> 红
        # (a)(1,0,0) + (1-a)(0.5,0.5,0.5)
        a = t
        # a **= 0.5
        return (0.5+0.5*a, 0.5-0.5*a, 0.5-0.5*a)

colors = np.array([blue_gray_red(t) for t in normed_scale], dtype=np.float32)

# 展开成三角形数据（position + normal + object_color）
data = []
for f in faces:
    n = get_normal(verts, *f)

    for idx in f:
        v = verts[idx]
        c = colors[idx]
        data.extend([*v, *n, *c])

data = np.array(data, dtype=np.float32)

# ====== 状态 ======
rot_x, rot_y = 0, 0
rot_x_light, rot_y_light = 0, 0
zoom = 2.5
left_down, right_down = False, False
last_x, last_y = 0, 0
last_x_light, last_y_light = 0, 0
wireframe = False

# ====== Shader ======
VERTEX_SHADER = """
#version 330
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 object_color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 FragPos;
out vec3 Normal;
out vec3 ObjectColor;

void main() {
    FragPos = vec3(model * vec4(position, 1.0));
    Normal = mat3(transpose(inverse(model))) * normal;
    ObjectColor = object_color;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330
in vec3 FragPos;
in vec3 Normal;
in vec3 ObjectColor;

out vec4 color;

uniform vec3 lightDir;

void main() {
    vec3 norm = normalize(Normal);
    float diff = max(dot(norm, normalize(lightDir)), 0.0);
    vec3 diffuse = diff * ObjectColor;
    vec3 ambient = 0.2 * ObjectColor;
    color = vec4(diffuse + ambient, 1.0);
}
"""

# ====== 初始化 ======
def init():
    global shader, VAO

    shader = compileProgram(
        compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
        compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    )

    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)

    glBindVertexArray(VAO)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)

    # position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 36, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    # normal
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 36, ctypes.c_void_p(12))
    glEnableVertexAttribArray(1)

    # object_color
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 36, ctypes.c_void_p(24))
    glEnableVertexAttribArray(2)

    glEnable(GL_DEPTH_TEST)

# ====== 矩阵工具 ======
def perspective(fov, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fov) / 2)

    mat = np.zeros((4,4), dtype=np.float32)
    mat[0,0] = f / aspect
    mat[1,1] = f
    mat[2,2] = (far + near) / (near - far)
    mat[2,3] = (2 * far * near) / (near - far)
    mat[3,2] = -1.0

    return mat

def lookAt(eye, center, up):
    f = center - eye
    f = f / np.linalg.norm(f)

    s = np.cross(f, up)
    s = s / np.linalg.norm(s)

    u = np.cross(s, f)

    mat = np.identity(4, dtype=np.float32)

    mat[0,0:3] = s
    mat[1,0:3] = u
    mat[2,0:3] = -f

    mat[0,3] = -np.dot(s, eye)
    mat[1,3] = -np.dot(u, eye)
    mat[2,3] =  np.dot(f, eye)

    return mat

def rotation_matrix(rx, ry):
    rx = np.radians(rx)
    ry = np.radians(ry)

    Rx = np.array([
        [1,0,0,0],
        [0,np.cos(rx),-np.sin(rx),0],
        [0,np.sin(rx),np.cos(rx),0],
        [0,0,0,1]
    ], dtype=np.float32)

    Ry = np.array([
        [np.cos(ry),0,np.sin(ry),0],
        [0,1,0,0],
        [-np.sin(ry),0,np.cos(ry),0],
        [0,0,0,1]
    ], dtype=np.float32)

    return Ry @ Rx

# ====== 绘制 ======
def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glUseProgram(shader)

    # 矩阵
    model = rotation_matrix(rot_x, rot_y)

    view = lookAt(
        np.array([0, 0, zoom], dtype=np.float32),
        np.array([0, 0, 0], dtype=np.float32),
        np.array([0, 1, 0], dtype=np.float32)
    )

    proj = perspective(45, 800 / 600, 0.1, 100)

    # 光方向（用右键控制）
    light_model = rotation_matrix(rot_x_light, rot_y_light)
    light_dir = light_model @ np.array([1,1,1,0], dtype=np.float32)

    # 传 uniform
    glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_TRUE, model)
    glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_TRUE, view)
    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_TRUE, proj)

    glUniform3f(glGetUniformLocation(shader, "lightDir"),
                light_dir[0], light_dir[1], light_dir[2])

    glBindVertexArray(VAO)
    glDrawArrays(GL_TRIANGLES, 0, len(data)//6)

    glutSwapBuffers()

# ====== 交互 ======
def mouse(button, state, x, y):
    global left_down, right_down, last_x, last_y, last_x_light, last_y_light, zoom

    if button == GLUT_LEFT_BUTTON:
        left_down = (state == GLUT_DOWN)
        last_x, last_y = x, y
    elif button == GLUT_RIGHT_BUTTON:
        right_down = (state == GLUT_DOWN)
        last_x_light, last_y_light = x, y

    if button == 3:
        zoom -= 0.2
    elif button == 4:
        zoom += 0.2

def motion(x, y):
    global rot_x, rot_y, rot_x_light, rot_y_light
    global last_x, last_y, last_x_light, last_y_light

    if left_down:
        rot_x += (y - last_y) * 0.5
        rot_y += (x - last_x) * 0.5
        last_x, last_y = x, y
    elif right_down:
        rot_x_light += (y - last_y_light) * 0.5
        rot_y_light += (x - last_x_light) * 0.5
        last_x_light, last_y_light = x, y

    glutPostRedisplay()

# ====== 键盘 ======
def keyboard(key, x, y):
    global wireframe

    if key == b'w':
        wireframe = not wireframe
        if wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    elif key == b'\x1b':  # ESC
        sys.exit()

# ====== 主函数 ======
def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    glutCreateWindow(b"Modern OpenGL Mesh")

    init()

    glutDisplayFunc(display)
    glutIdleFunc(display)
    glutMouseFunc(mouse)
    glutMotionFunc(motion)
    glutKeyboardFunc(keyboard)

    glutMainLoop()

if __name__ == "__main__":
    main()

