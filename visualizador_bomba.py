import meshio
from pathlib import Path

# /home/vzujo/miniconda3/envs/fenics-env/bin/python /home/vzujo/Codes/ards-lung-simulator-copy-agu/visualizador_bomba.py

files = sorted(Path("outputs/PIG5-mc-per/VTK").glob("*.vtu"))
print(f"Found {len(files)} .vtu files.")

for f in files:
    #f = files[1]
    mesh = meshio.read(f)

    print(f"Mesh info for {f}:")

    print("Point data:")
    for name, data in mesh.point_data.items():
        print(f"    {name}, shape: {data.shape}")

    print("Cell data:")
    for name, data in mesh.cell_data.items():
        print(f"    {name}")

    print(f"{mesh.field_data = }")