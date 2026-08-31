## Extra demo data — 16 demos, edge midpoints

- Order: orientation records (0° → 45° → 90° → 135°), within each orientation 4 edge midpoints starting at the top edge and going clockwise (top → right → bottom → left).
- Episode k (0-indexed): orientation = [0,45,90,135][k // 4], position = [top, right, bottom, left][k % 4]
- Positions: midpoint of each edge of the start region (halfway between adjacent eval corners), ≥ ~half an edge length from any eval cell. Same collection protocol, reset ritual, lighting, and task string as for original demo dataset.