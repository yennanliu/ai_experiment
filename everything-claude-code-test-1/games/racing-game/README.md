# Racing Car Game

A simple, fun top-down vertical scrolling racing game built with vanilla HTML5, CSS, and JavaScript.

## How to Play

1. **Start the Game**: Simply open `index.html` in any modern web browser
2. **Begin Racing**: Press `SPACE` to start
3. **Avoid Obstacles**: Dodge the incoming cars for as long as possible
4. **Survive**: The longer you survive, the higher your score!

## Controls

- **Arrow Left (←)**: Move car left
- **Arrow Right (→)**: Move car right
- **Arrow Up (↑)**: Increase speed (optional)
- **Arrow Down (↓)**: Decrease speed (optional)
- **SPACE**: Start game / Restart after game over

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

- Smooth 60 FPS gameplay
- Progressive difficulty system
- High score persistence (saved in browser localStorage)
- Simple, intuitive controls
- Responsive keyboard input
- Visual collision feedback
- Clean, modern UI

## Technical Details

- **Canvas Size**: 800x600 pixels
- **Road**: 3 lanes with animated lane markers
- **Player Car**: 40x60 pixels, controllable speed
- **Obstacles**: Random spawn positions, variable speeds
- **Collision Detection**: Rectangle-based (AABB)
- **Performance**: Optimized for 60 FPS

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

This game is built with:
- HTML5 Canvas for rendering
- Vanilla JavaScript (no dependencies)
- CSS3 for styling
- requestAnimationFrame for smooth animation

No build process or installation required - just open and play!

---

Enjoy the game and try to beat your high score! 🏎️
