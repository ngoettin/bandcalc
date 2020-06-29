# band-calc
Calculate and plot band structures for different lattices

## Build documentation

 * `cd docs`
 * `make html`
 * `firefox _build/html/index.html` (or any other browser)

## Examples
```
python -m bandcalc.examples.bandstructure.hexagonal
```
![image of band structure](example_images/hexagonal_band_structure.png?raw=true)

```
python -m bandcalc.examples.misc.monkhorst_pack
```
![image of monkhorst pack lattice](example_images/monkhorst_pack_lattice.png?raw=true)

```
python -m bandcalc.examples.potential.moire_reciprocal
```
![image of moire potentials](example_images/moire_potential.png?raw=true)

```
python -m bandcalc.examples.bandstructure.moire_hexagonal --potential MoS2 --angle 3
```
![image of moire potentials](example_images/moire_bandstructure.png?raw=true)
