# Reunión con Agustín sobre super-codigo

## Estrutura

### Estructura completa

- raw-data/
- testing-data/
- src/
    - analysis/
    - calibrate/
    - execution/
    - legacy/
    - meshing/
    - signal-processing/
    - test/
    - utilities/
    - AirwayHelper.py
    - AirwayManager.py
    - AwMngr_workbench.py
    - functions2.py
    - modelfunctions.py
    - simple-skel-generator.py
    - suppoertfunctions.py
    - vcv_lung.py
- README

### Estructura relevante

- raw-data/
- testing-data/
- src/
    - analysis/
    - execution/
        - 2025-12-universal_executer.py
    - meshing/
        - 1_initial_meshing.py
        - 2_extrabc_info.py
        - 3_final_meshing_wbc.py
    - utilities/
    - AirwayHelper.py
    - AirwayManager.py
    - modelfunctions.py
    - suppoertfunctions.py
    - vcv_lung.py
- README


## src

Es la carpeta principal, el resto es casi basura.

### Analysis

- Los que no tienen fecha son super antiguos
- también contiene tareas random que pide le profe

### Calibrate

- No lo debería ocupar, creo que el BOCalibration es el más o único que se ocupa
- El resto es probable que se eliminen

### Execution

- 2025-12-universal_executer es el que recopila todo mas o menos, el resto podría ser chatarra; voy a trabajar en este posiblemente.

### Meshing

- Estan los 1, 2, 3, el resto es chatarra (más que chatarra es código que se pudo simplificar)

### SignalProcessing

- No lo debería pescar para mi trabajo; le sirve más a agustín

### Tests

- también es chatarra. Lo único relevante podría ser el inverse-problem-lab; sirve para aprender a hacer problemas inversos, pero es probable que nunca lo use.

### Utilities

- Son funciones aisladas creo.

### Archivos sueltos

Todo lo que está fuera de src es relevante
- Airwayhelper es porbable que sea chatarra, pero puedo revisar el código
- AirwayManager es super útil, tiene todo lo que se trata de airways; es oro puro pero requiere de juntarse toda la tarde con agustín.

En la clase Tree está oro puro, y si quisera implementar lo de Bates con resistencia infinita, sería aquí.

- Airway_workbench: es como un lugar de prueba similar al último del código anterior

- functions2.py: es chatarra

- modelfunctions.py: super importante; en principio no debería editarlo; está todo en fenicsx

- simple-skel-generator: chatarra; construye un arbol sintético asiq puede ser útil.

- suppoerfunctions: de nivaldo. Lo importante es que se define una función que define el solver (es un wrapper)

- manage_output_directory: maneja donde mandar los archivos.
- retrieve_last_checkpoint: guarda checkpoints después de cada iteración
- resume_checkpoint: lo que dice.
(Estos están rotos por ahora, pero son super útiles)

vcv_lung: Este es el código real. hay 2 fnciones de support
- execute_vcv_simularion: esta es la grande super bomba función para hacer la simulación completa.



## To-do

- Partir también revisando los códigos de Nivaldo
- Agendar una reu con hurtado para contarle sólo que me reuní con Agustín y quizás corrí una simulación
- correr el 2025-12_universal-executer.py con -pig_id=2 - mesh_type=medium
- me mandara 


### Tips

- Trabaja con la geometría más penca que te permita sacar resultados rápido
- Recién para el paper le subes la resolución
