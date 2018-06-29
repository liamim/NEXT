from pathlib import Path
import networkx as nx
import yaml

G = nx.grid_2d_graph(21, 21)
filepaths = []
neighbors = []
locations = []
for i, p in enumerate(Path('.').glob('cat_dog/*.png')):
    x, y = map(int, p.stem.split('_'))
    n = (x+10, y+10)
    neighbors.append(list(G.neighbors(n)))
    G.nodes[n]['i'] = i
    locations.append((x, y))
    filepaths.append(p)

targets = [
    {'primary_description': f'http://localhost:8889/{p.name}',
     'primary_type': 'image',
     'alt_description': '',
     'alt_type': '',
     # 'neighbors_': ns,
     'location': loc,
     'neighbors': [G.nodes[n]['i'] for n in ns],
    }
    for p, ns, loc in zip(filepaths, neighbors, locations)
]

print(yaml.safe_dump(targets))
