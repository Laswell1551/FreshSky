# Map data license and provenance

Building data © OpenStreetMap contributors.

- Attribution: https://www.openstreetmap.org/copyright
- Database license: Open Database License (ODbL) 1.0
- License text: https://opendatacommons.org/licenses/odbl/1-0/

The cached OpenStreetMap JSON databases, the converted map geometry, and derived geometric trace databases under data/ retain ODbL 1.0. Their distribution is separate from the MIT-licensed experiment code. Retain this attribution and license notice when sharing those data.

The cached regions are:

| Cache | Bounding box (south, west, north, east) |
|---|---|
| osm_midtown_manhattan.json | 40.7485, -73.9911, 40.7593, -73.9769 |
| osm_city_of_london.json | 51.5100, -0.0900, 51.5200, -0.0740 |

Height conversion prefers explicit height, then building:levels × 3 m, then a 10-m default, clipped to 2–500 m. The geometry grid has 6-m cells. These are public city maps and simulated patrol paths, not tracked flights or measured radio links.

Primary traces osm_patrol_*_geo1.npz through geo5.npz contain LoS flags, link distances, sampled route positions, assigned speeds, and height-grid clearances. The route generator and ray-marching code are in scripts/city_multitrajectory.py and scripts/geometry_builder.py. The legacy osm_traces_*.npz files support the original single wrapped-trajectory stress test.
