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
