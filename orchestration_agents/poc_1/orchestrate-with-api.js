#!/usr/bin/env node

/**
 * Fully Automated Orchestration using Claude API
 *
 * This script runs all 3 agents (PM, Backend, Frontend) automatically
 * using the Claude API with no manual intervention required.
 *
 * Usage: node orchestrate-with-api.js "Your feature request"
 * Example: node orchestrate-with-api.js "Add dark mode toggle"
 */

import Anthropic from '@anthropic-ai/sdk';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration
const CONFIG = {
  model: 'claude-sonnet-4-5-20250929',
  maxTokens: 4096,
  workspaceDir: path.join(__dirname, 'workspace'),
  agentsDir: path.join(__dirname, '.claude/agents')
};

// Colors for terminal output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  green: '\x1b[32m',
  blue: '\x1b[34m',
  yellow: '\x1b[33m',
  red: '\x1b[31m',
  cyan: '\x1b[36m'
};

function log(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

function logSection(title) {
  log(`\n${'━'.repeat(50)}`, 'cyan');
  log(`  ${title}`, 'bright');
  log('━'.repeat(50), 'cyan');
}

// Initialize Claude API client
function createClient() {
  const apiKey = process.env.ANTHROPIC_API_KEY;

  if (!apiKey) {
    log('❌ Error: ANTHROPIC_API_KEY environment variable not set', 'red');
    log('\nSet it with:', 'yellow');
    log('  export ANTHROPIC_API_KEY="your-api-key-here"', 'cyan');
    log('\nOr create a .env file:', 'yellow');
    log('  echo "ANTHROPIC_API_KEY=your-api-key" > .env', 'cyan');
    process.exit(1);
  }

  return new Anthropic({ apiKey });
}

// Read agent definition file
async function readAgent(agentName) {
  const agentPath = path.join(CONFIG.agentsDir, `${agentName}.md`);
  try {
    const content = await fs.readFile(agentPath, 'utf-8');
    return content;
  } catch (error) {
    log(`❌ Failed to read agent file: ${agentPath}`, 'red');
    throw error;
  }
}

// Call Claude API with agent system prompt
async function runAgent(client, agentPrompt, userMessage, agentName) {
  log(`\n🤖 Running ${agentName} Agent...`, 'blue');
  log(`   Thinking...`, 'yellow');

  try {
    const response = await client.messages.create({
      model: CONFIG.model,
      max_tokens: CONFIG.maxTokens,
      system: agentPrompt,
      messages: [
        {
          role: 'user',
          content: userMessage
        }
      ]
    });

    const result = response.content[0].text;
    log(`   ✓ ${agentName} Agent completed`, 'green');

    // Log token usage
    const inputTokens = response.usage.input_tokens;
    const outputTokens = response.usage.output_tokens;
    log(`   📊 Tokens: ${inputTokens} in / ${outputTokens} out`, 'cyan');

    return result;
  } catch (error) {
    log(`   ❌ Error running ${agentName} Agent: ${error.message}`, 'red');
    throw error;
  }
}

// Save agent output to file
async function saveOutput(filename, content) {
  const filePath = path.join(CONFIG.workspaceDir, filename);
  await fs.writeFile(filePath, content, 'utf-8');
  log(`   💾 Saved: ${filename}`, 'green');
}

// Main orchestration flow
async function orchestrate(featureRequest) {
  logSection('🚀 Automated Agent Orchestration');
  log(`Feature Request: ${featureRequest}`, 'bright');

  const startTime = Date.now();
  const client = createClient();

  try {
    // Save feature request
    await saveOutput('feature-request.txt', featureRequest);

    // ======================
    // AGENT 1: PM Agent
    // ======================
    logSection('📋 Phase 1: Product Manager Agent');

    const pmAgent = await readAgent('pm');
    const pmContext = `Feature Request: ${featureRequest}

Please analyze this feature request and generate detailed requirements following the template structure defined in your role. Output the complete requirements document in markdown format.`;

    const pmOutput = await runAgent(client, pmAgent, pmContext, 'PM');
    await saveOutput('feature-requirements.md', pmOutput);

    // ======================
    // AGENT 2: Backend Agent
    // ======================
    logSection('⚙️  Phase 2: Backend Engineer Agent');

    const backendAgent = await readAgent('backend');
    const backendContext = `Feature Request: ${featureRequest}

Requirements from PM Agent:
${pmOutput}

Please analyze if this feature requires any backend API changes. Generate a detailed API change specification in JSON format following your template structure.`;

    const backendOutput = await runAgent(client, backendAgent, backendContext, 'Backend');
    await saveOutput('feature-api-changes.json', backendOutput);

    // ======================
    // AGENT 3: Frontend Agent
    // ======================
    logSection('💻 Phase 3: Frontend Engineer Agent');

    const frontendAgent = await readAgent('frontend');
    const frontendContext = `Feature Request: ${featureRequest}

Requirements from PM Agent:
${pmOutput}

API Analysis from Backend Agent:
${backendOutput}

Please design the frontend implementation for this feature following your template structure. Provide a detailed implementation plan in markdown format.`;

    const frontendOutput = await runAgent(client, frontendAgent, frontendContext, 'Frontend');
    await saveOutput('feature-ui-implementation.md', frontendOutput);

    // ======================
    // AGENT 4: Implementation Agent
    // ======================
    logSection('🔨 Phase 4: Implementation Agent');

    const implementerContext = `You are an expert full-stack developer. Based on the specifications below, implement the complete feature.

Feature Request: ${featureRequest}

PM Requirements:
${pmOutput}

Backend Analysis:
${backendOutput}

Frontend Implementation Plan:
${frontendOutput}

Please provide:
1. Complete, production-ready code for all necessary files
2. Clear file paths for each code block
3. Step-by-step implementation instructions
4. Any configuration changes needed

Format your response as a detailed implementation guide with code blocks for each file that needs to be created or modified.`;

    const implementationOutput = await runAgent(client, implementerContext, implementerContext, 'Implementation');
    await saveOutput('feature-implementation-code.md', implementationOutput);

    // ======================
    // Summary
    // ======================
    const duration = ((Date.now() - startTime) / 1000).toFixed(2);

    logSection('✅ Orchestration Complete!');
    log(`\n📁 Generated Files:`, 'bright');
    log(`   ✓ workspace/feature-request.txt`, 'green');
    log(`   ✓ workspace/feature-requirements.md`, 'green');
    log(`   ✓ workspace/feature-api-changes.json`, 'green');
    log(`   ✓ workspace/feature-ui-implementation.md`, 'green');
    log(`   ✓ workspace/feature-implementation-code.md`, 'green');

    log(`\n⏱️  Total Time: ${duration}s`, 'cyan');
    log(`\n💡 Next Steps:`, 'yellow');
    log(`   1. Review the implementation code in workspace/feature-implementation-code.md`, 'cyan');
    log(`   2. Apply the code changes to your project`, 'cyan');
    log(`   3. Test in your running application at http://localhost:5173`, 'cyan');

    log('\n' + '━'.repeat(50), 'cyan');

  } catch (error) {
    log(`\n❌ Orchestration failed: ${error.message}`, 'red');
    process.exit(1);
  }
}

// CLI Entry Point
if (process.argv.length < 3) {
  log('Usage: node orchestrate-with-api.js "Your feature request"', 'yellow');
  log('\nExamples:', 'cyan');
  log('  node orchestrate-with-api.js "Add dark mode toggle"', 'bright');
  log('  node orchestrate-with-api.js "Add export notes to PDF"', 'bright');
  log('  node orchestrate-with-api.js "Add note favorites/pinning"', 'bright');
  log('\nMake sure to set ANTHROPIC_API_KEY environment variable!', 'yellow');
  process.exit(1);
}

const featureRequest = process.argv.slice(2).join(' ');
orchestrate(featureRequest);
