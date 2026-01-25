# Racing Car Game Implementation Plan

## Overview
Create a simple top-down vertical scrolling racing car game using vanilla HTML5, CSS, and JavaScript. The game will be self-contained in a single HTML file for maximum simplicity - just open in a browser to play.

## Game Concept
- **Type**: Top-down vertical scrolling racer
- **Controls**: Arrow keys (left/right to move, optional up/down for speed)
- **Objective**: Avoid obstacles (other cars) and survive as long as possible
- **Difficulty**: Progressive - obstacles spawn faster and move quicker over time
- **Visuals**: Simple geometric shapes (rectangles/triangles) rendered on HTML5 Canvas

## Critical Files to Create

1. **`/Users/jerryliu/ai_experiment/everything-claude-code-test-1/games/racing-game/index.html`**
   - Main game file with embedded CSS and JavaScript
   - HTML5 Canvas for game rendering (800x600px)
   - UI overlays for start screen, game over, and score display

2. **`/Users/jerryliu/ai_experiment/everything-claude-code-test-1/games/racing-game/README.md`**
   - How to play instructions
   - Controls reference
   - Game rules

## Implementation Steps

### 1. Setup Foundation
- Create `games/racing-game/` directory structure
- Build `index.html` with:
  - HTML5 canvas element
  - Embedded CSS for layout and styling
  - Basic JavaScript structure with game configuration

### 2. Game Loop & Canvas Setup
- Initialize canvas context (2D)
- Implement main game loop using `requestAnimationFrame`
- Set up game state management (MENU, PLAYING, GAMEOVER)
- Create frame-independent movement using deltaTime

### 3. Player Car
- Create player object with position (x, y), dimensions, and speed
- Implement keyboard input handling (arrow keys)
- Add smooth left/right movement
- Enforce road boundaries (keep car on road)
- Render player car as colored rectangle/triangle

### 4. Scrolling Road
- Draw road background with lanes
- Create animated lane markers that scroll downward
- Add road edges and grass/shoulder areas
- Loop marker positions for infinite scrolling effect

### 5. Obstacle System
- Create obstacle objects (enemy cars) with:
  - Random spawn positions (within lanes)
  - Variable speeds
  - Different colors
- Implement spawning logic at timed intervals
- Update obstacle positions (move down screen)
- Remove off-screen obstacles to manage memory
- Render obstacles as colored rectangles

### 6. Collision Detection
- Implement rectangle-based collision detection (AABB)
- Check player vs all obstacles each frame
- On collision:
  - Trigger game over state
  - Stop game loop
  - Display final score
  - Show restart option

### 7. Scoring & Game States
- Track score based on time survived and distance traveled
- Display score continuously during gameplay
- Store high score in localStorage
- Implement state transitions:
  - Start screen: "Press SPACE to Start" + instructions
  - Playing: Active gameplay with score display
  - Game Over: Show final score, high score, restart option

### 8. Progressive Difficulty
- Gradually increase obstacle spawn rate over time
- Incrementally increase obstacle speed
- Add more obstacles on screen simultaneously as game progresses
- Balance difficulty curve for 2-5 minute engaging gameplay

### 9. Polish & Documentation
- Add visual feedback (screen flash on collision)
- Smooth animations and transitions
- Create README.md with game instructions
- Test in multiple browsers (Chrome, Firefox, Safari)
- Verify performance (maintain 60 FPS)

## Key Technical Details

### Game Configuration
```javascript
canvas: 800x600px
player: 40x60px, speed 5px/frame
road: 400px wide, 3 lanes
obstacles: spawn every 1-2 seconds, speed 3-7px/frame
```

### Architecture
- Single-file approach with embedded CSS/JS
- Canvas-based rendering
- Event-driven input handling
- Object-oriented game entities (player, obstacles)
- State machine for game flow

### Core Game Elements
1. **Player Car**: Bottom-center starting position, arrow key controls
2. **Road**: Continuous scrolling with lane markers
3. **Obstacles**: Enemy cars spawning from top
4. **Collision System**: Rectangle intersection detection
5. **Score Display**: Real-time overlay
6. **Game States**: Menu → Playing → Game Over

## Verification Steps

After implementation, verify:

1. **Functionality**
   - Open `games/racing-game/index.html` in browser
   - Press SPACE to start game
   - Use arrow keys to move car left/right
   - Verify car stays within road boundaries
   - Check obstacles spawn and scroll down
   - Confirm collision ends game
   - Verify score increases during gameplay
   - Test restart functionality after game over

2. **Performance**
   - Smooth 60 FPS animation
   - No lag during obstacle spawning
   - Responsive keyboard controls (no input delay)

3. **Difficulty Progression**
   - Game starts easy with few obstacles
   - Difficulty increases noticeably after 30-60 seconds
   - Remains challenging but fair

4. **Browser Compatibility**
   - Test in Chrome, Firefox, Safari
   - Verify canvas rendering works correctly
   - Check localStorage persists high score

5. **User Experience**
   - Clear instructions on start screen
   - Intuitive controls
   - Visible score display
   - Obvious game over state
   - Easy restart process

## Success Criteria

✓ Game loads instantly in browser (no build required)
✓ Smooth 60 FPS gameplay
✓ Responsive controls with no noticeable lag
✓ Fair collision detection
✓ Progressive difficulty keeps game engaging
✓ High score persists across sessions
✓ Clear visual feedback for all game events
✓ Works in modern browsers without issues

## Future Enhancements (Optional)

- Sound effects and background music
- Different car types with varied stats
- Power-ups (shields, speed boosts)
- Multiple environments (city, desert, snow)
- Touch controls for mobile
- Improved graphics with sprites/images
- Particle effects for collisions
- Additional game modes
