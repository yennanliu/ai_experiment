# Racing Car Game

Two exciting versions of a racing game: a classic 2D top-down racer and an immersive 3D racing experience!

## Game Versions

### 2D Version (`index.html`)
A simple, fun top-down vertical scrolling racing game built with vanilla HTML5 Canvas, CSS, and JavaScript.

### 3D Version (`index-3d.html`)
An immersive 3D racing experience built with Three.js, featuring realistic 3D graphics, camera following, and depth perception.

## How to Play

**For 2D Version:**
1. Open `index.html` in any modern web browser
2. Press `SPACE` to start
3. Dodge the incoming cars using arrow keys
4. Survive as long as possible!

**For 3D Version:**
1. Open `index-3d.html` in any modern web browser
2. Press `SPACE` to start
3. Experience 3D racing with immersive graphics
4. Control your speed and steering to avoid obstacles!

## Controls

**Both Versions:**
- **Arrow Left (←)**: Steer left
- **Arrow Right (→)**: Steer right
- **Arrow Up (↑)**: Increase speed
- **Arrow Down (↓)**: Decrease speed
- **SPACE**: Start game / Restart after game over

**Note:** In the 3D version, speed control is more prominent with a speedometer display, while the 2D version treats speed as optional.

## Game Rules

- Stay on the road and avoid colliding with other cars
- Your score increases the longer you survive
- The game gets progressively harder:
  - Obstacles spawn more frequently
  - Obstacles move faster
  - More cars appear on screen
- Your high score is automatically saved in your browser
- Collision with any obstacle ends the game

## Objective

Survive as long as possible while avoiding obstacles to achieve the highest score!

## Features

**Both Versions:**
- Smooth 60 FPS gameplay
- Progressive difficulty system
- High score persistence (saved in browser localStorage)
- Simple, intuitive controls
- Responsive keyboard input
- Clean, modern UI

**3D Version Additional Features:**
- Immersive 3D graphics with Three.js
- Dynamic camera following player
- Realistic shadows and lighting
- 3D car models with wheels and details
- Speedometer display
- Depth perception for better gameplay
- Car tilts when steering
- Beautiful sky and grass environment

## Technical Details

### 2D Version
- **Canvas Size**: 800x600 pixels
- **Rendering**: HTML5 Canvas 2D Context
- **Road**: 3 lanes with animated lane markers
- **Player Car**: 40x60 pixels, controllable speed
- **Collision Detection**: Rectangle-based (AABB)
- **Performance**: Optimized for 60 FPS

### 3D Version
- **Rendering**: Three.js (WebGL)
- **3D Models**: Geometric primitives (boxes, cylinders)
- **Camera**: Perspective camera with follow system
- **Lighting**: Ambient + Directional with shadows
- **Road**: 12 units wide, 3 lanes, infinite scrolling
- **Physics**: 3D collision detection with depth
- **Performance**: Hardware-accelerated 60 FPS

## Browser Compatibility

Works on all modern browsers:
- Chrome
- Firefox
- Safari
- Edge

## Tips for High Scores

1. Stay in the center lane when possible for more escape routes
2. Use the speed controls (up/down arrows) strategically
3. Focus on smooth movements rather than rapid corrections
4. Watch the pattern of incoming cars to plan your moves
5. Don't panic - the game rewards calm, calculated movements

## Development

### 2D Version Built With:
- HTML5 Canvas for rendering
- Vanilla JavaScript (no dependencies)
- CSS3 for styling
- requestAnimationFrame for smooth animation

### 3D Version Built With:
- Three.js (r128) for 3D rendering via CDN
- WebGL for hardware acceleration
- Vanilla JavaScript for game logic
- CSS3 for UI styling
- Shadow mapping for realistic lighting

No build process or installation required for either version - just open and play!

---

Enjoy the game and try to beat your high score! 🏎️
