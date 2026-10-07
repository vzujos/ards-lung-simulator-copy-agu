# Reunión con Agustín sobre super-codigo

## Estrutura (relevante)

```
ards-lung-simulator/
    src/
        core/
            AirwayManager.py
            modelfunctions.py
            supportfunctions.py
            vcv_lung.py
        experiment/
            AirwayHelper.py
            AwMngr_workbench.py
        meshing/
            1_initial_meshing.py
            2_extrabc_info.py
            3_final_meshing_wbc.py
        postprocessing/
            simulation-vtu-grouper.py
    scripts/
        universal-executer.py
    results-data/
    testing-data/
    README
```

### Archivos relevantes

Todo lo que está fuera de src es relevante

- Airwayhelper es probable que sea chatarra, pero puedo revisar el código.
- AirwayManager es super útil, tiene todo lo que se trata de airways; es oro puro pero requiere de juntarse toda la tarde con agustín. En la clase Tree está oro puro, y si quisiera implementar lo de Bates con resistencia infinita, sería aquí.
- supportfunctions: de nivaldo. Lo importante es que se define una función que define el solver (es un wrapper)
- vcv_lung: Este es el código real. hay 2 funciones de support
- execute_vcv_simularion: esta es la grande super bomba función para hacer la simulación completa.


### Codigo corrido 1

- Medium-coarse. T0=20:10. Tf=23:55 DeltaT = 13472s = 3:45 h
- Se guardan los resultados crudos en results-data/Pigs5-mc-per-1/
- Se guardan los resultados procesados en outputs/Pig5-mc-per/post/

### Cambios

- Se actualizó el código `manuscript/code/generate_pig5_figures.py` para generar las figuras.
- Se agregó el campo delta_HU_clasif en Paraview que muestra el cambio de clasificación, pero por alguna razón es distinto al obtenido desde Python.