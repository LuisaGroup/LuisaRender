from msilib.schema import Error
import os
import sys

v_count_duplicated = 0
f_count_duplicated = 0
vertex_set = set()
face_set = set()

def obj_analyse(obj_file):
    """
    Analyse the obj file and return the vertex and face set.
    """
    global v_count_duplicated, f_count_duplicated
    global vertex_set, face_set

    with open(obj_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith('v '):
            vertex_set.add(line.strip('\n'))
            v_count_duplicated += 1
        elif line.startswith('f '):
            faces = line.split(' ')[1:]

            def load_mesh(faces):
                global f_count_duplicated
                global face_set
                mesh = ''
                for face in faces:
                    face = face.split('/')[0]
                    mesh += lines[int(face)].strip('\n') + '\n'
                face_set.add(mesh)
                f_count_duplicated += 1

            if len(faces) == 3:
                load_mesh(faces)
            elif len(faces) == 4:
                load_mesh(faces[0:3])
                load_mesh(faces[1:4])
            else:
                raise Error('Invalid face count: {}'.format(len(faces)))

if __name__ == '__main__':
    argc = len(sys.argv)
    if argc != 2:
        print('Usage: obj-analyse.py <obj_file/obj_contents>')
        exit(1)

    path = sys.argv[1]
    if os.path.isfile(path):
        obj_analyse(path)
    else:
        files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.obj')]
        print(len(files))
        for f in files:
            obj_analyse(f)

    print('Vertex count: {}'.format(len(vertex_set)))
    print('Face count: {}'.format(len(face_set)))
    print('Vertex duplicated count: {}'.format(v_count_duplicated))
    print('Face duplicated count: {}'.format(f_count_duplicated))
    # print('Vertex: {}'.format(vertex_set))
    # print('Face: {}'.format(face_set))
