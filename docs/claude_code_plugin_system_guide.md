# Claude Code Plugin System: Complete Reference

Based on official documentation from https://code.claude.com/docs/

---

## 1. Plugin Directory Structure

### Exact Required Layout

```
my-plugin/
├── .claude-plugin/
│   └── plugin.json              # ONLY this goes in .claude-plugin/
├── skills/
│   ├── skill-name/
│   │   ├── SKILL.md             # Required: describes the skill
│   │   ├── reference.md         # Optional supporting files
│   │   └── scripts/             # Optional scripts
│   └── another-skill/
│       └── SKILL.md
├── commands/                    # Flat markdown skill files (legacy)
│   ├── deploy.md
│   └── status.md
├── agents/                      # Subagent definitions
│   ├── security-reviewer.md
│   └── compliance-checker.md
├── hooks/
│   └── hooks.json               # Hook event handlers
├── .mcp.json                    # MCP server configs
├── .lsp.json                    # Language server configs
├── monitors/
│   └── monitors.json            # Background monitor configs
├── output-styles/              # Output style definitions
├── themes/                      # Color themes (experimental)
├── bin/                         # Executables added to PATH
├── settings.json                # Default plugin settings
└── README.md                    # Documentation
```

### Critical Rule
**DO NOT** put `skills/`, `commands/`, `agents/`, `hooks/`, or any component directories inside `.claude-plugin/`. 
Only `plugin.json` belongs in `.claude-plugin/`.

---

## 2. Plugin Manifest Schema (`.claude-plugin/plugin.json`)

### Complete Schema Example

```json
{
  "name": "my-plugin",
  "displayName": "My Plugin",
  "version": "1.2.0",
  "description": "Brief description of what the plugin does",
  "author": {
    "name": "Your Name",
    "email": "you@example.com",
    "url": "https://github.com/yourname"
  },
  "homepage": "https://docs.example.com",
  "repository": "https://github.com/yourname/plugin",
  "license": "MIT",
  "keywords": ["keyword1", "keyword2"],
  "defaultEnabled": true,
  
  "skills": "./custom/skills/",
  "commands": ["./commands/"],
  "agents": ["./agents/"],
  "hooks": "./hooks/hooks.json",
  "mcpServers": "./.mcp.json",
  "outputStyles": "./styles/",
  "lspServers": "./.lsp.json",
  
  "userConfig": {
    "api_endpoint": {
      "type": "string",
      "title": "API endpoint",
      "description": "Your API endpoint",
      "required": true
    },
    "api_token": {
      "type": "string",
      "title": "API token",
      "description": "Authentication token",
      "sensitive": true
    }
  },
  
  "dependencies": [
    "helper-lib",
    { "name": "secrets-vault", "version": "~2.1.0" }
  ],
  
  "experimental": {
    "themes": "./themes/",
    "monitors": "./monitors.json"
  }
}
```

### Required Fields (if manifest is present)
- **`name`** (string, kebab-case): Unique identifier used for namespacing. Skills become `/name:skillname`.

### Important Optional Fields

| Field | Type | Purpose | Notes |
|-------|------|---------|-------|
| `version` | string | Semantic version (e.g., "1.2.0") | Omit to use git commit SHA; every commit = new version |
| `description` | string | Shown in `/plugin` manager | Required for discoverability |
| `author` | object | Author info (`name` required, `email` optional) | Attribution |
| `displayName` | string | Human-readable name (with spaces allowed) | v2.1.143+; used in UI only |
| `defaultEnabled` | boolean | Plugin auto-enabled on install? (default: true) | v2.1.154+ |
| `skills` | string\|array | Custom skill directories | Defaults to `skills/`; can extend with custom paths |
| `commands` | string\|array | Flat markdown skill files | Defaults to `commands/`; replaces default if set |
| `agents` | string\|array | Custom agent files | Defaults to `agents/`; replaces if set |
| `hooks` | string\|array\|object | Hook config paths or inline | Can mix file paths and inline objects |
| `mcpServers` | string\|array\|object | MCP server configs | File paths or inline JSON |
| `lspServers` | string\|array\|object | LSP server configs | File paths or inline JSON |
| `userConfig` | object | Config values prompted at enable time | Keys available as `${user_config.KEY}` in MCP/LSP/hooks |
| `dependencies` | array | Plugin dependencies with optional version constraints | Semver ranges: `~2.1.0`, `^2.0.0` |

### Path Resolution Rules
- All paths must be relative to plugin root and start with `./`
- **Replaces default**: `commands`, `agents`, `outputStyles`, `experimental.themes`, `experimental.monitors`
- **Extends default**: `skills` (always scans `skills/` + custom paths)
- **Own rules**: `hooks`, `mcpServers`, `lspServers` (merge from multiple sources)

### Manifest is Optional
If omitted, Claude Code:
- Auto-discovers components in default locations
- Derives plugin name from directory name
- Uses git commit SHA as version

---

## 3. Skill Format Inside Plugins

### Location
`skills/<skill-name>/SKILL.md` (directory with markdown file inside)

### Frontmatter Fields
```yaml
---
description: "What this skill does. Used for skill discovery."
disable-model-invocation: false  # Set true if skill is manual-only
allowed-tools: []               # Restrict which tools skill can use (optional)
---
```

### Naming & Invocation
- Directory name becomes skill name: `skills/code-review/` → `/plugin-name:code-review`
- Or use frontmatter `name` field to override: `name: "my-alias"` → `/plugin-name:my-alias`
- Supports `$ARGUMENTS` placeholder for user input: `/plugin-name:skill-name Alex`

### Example
```markdown
skills/code-review/SKILL.md
---
description: Reviews code for best practices, bugs, and security issues
---

Review the selected code for:
1. Potential bugs or edge cases
2. Security concerns
3. Performance issues
4. Readability and maintainability

Be concise and actionable. Suggest specific fixes where applicable.
```

---

## 4. Marketplace Structure

### `marketplace.json` Schema

Create `.claude-plugin/marketplace.json` in your marketplace repository root.

```json
{
  "name": "my-plugins",
  "owner": {
    "name": "Your Team",
    "email": "team@example.com"
  },
  "description": "Our team's collection of plugins",
  "version": "1.0.0",
  
  "metadata": {
    "pluginRoot": "./plugins"  # Base dir prepended to relative paths
  },
  
  "plugins": [
    {
      "name": "code-formatter",
      "source": "./plugins/formatter",  # Relative path, GitHub repo, or object
      "description": "Auto-format code on save",
      "version": "2.1.0",
      "author": { "name": "Dev Team" },
      "category": "productivity",
      "tags": ["formatting", "code-style"],
      "defaultEnabled": false
    },
    {
      "name": "deployment-tools",
      "source": {
        "source": "github",
        "repo": "company/deploy-plugin",
        "ref": "main",
        "sha": "a1b2c3d4..."  # Exact commit pin
      }
    }
  ]
}
```

### Required Fields
- **`name`** (string, kebab-case): Marketplace identifier (public-facing, used in `/plugin install plugin@marketplace-name`)
- **`owner`** (object): Marketplace maintainer (`name` required, `email` optional)
- **`plugins`** (array): List of plugins

### Optional Marketplace Fields
| Field | Type | Purpose |
|-------|------|---------|
| `description` | string | Marketplace description for users |
| `version` | string | Marketplace manifest version |
| `metadata.pluginRoot` | string | Base dir for relative plugin paths |
| `allowCrossMarketplaceDependenciesOn` | array | Other marketplaces this can depend on |

### Plugin Entry Fields (inside `plugins` array)

**Required:**
- `name` (string, kebab-case)
- `source` (string or object) — see sources below

**Optional:**
- All fields from plugin manifest schema (description, version, author, homepage, repository, license, keywords, category, tags, defaultEnabled, etc.)
- `strict` (boolean): Controls if `plugin.json` or marketplace entry is authority (default: true = `plugin.json` is authority)

### Plugin Sources

| Source Type | Object Format | Notes |
|------------|---------------|----- |
| **Relative path** | `"./plugins/my-plugin"` | Local dir in marketplace repo; must start with `./` |
| **GitHub** | `{"source": "github", "repo": "owner/repo", "ref": "v1.0", "sha": "..."}` | GitHub shorthand works |
| **Git URL** | `{"source": "url", "url": "https://gitlab.com/team/plugin.git", "ref": "main"}` | Any git host |
| **Git subdir** | `{"source": "git-subdir", "url": "...", "path": "tools/plugin", "ref": "..."}` | Subdirectory in a git repo |
| **npm package** | `{"source": "npm", "package": "@org/plugin", "version": "^2.0.0", "registry": "..."}` | npm registry |

---

## 5. Creating a Local Marketplace

### Step-by-Step

1. **Create directory structure:**
   ```bash
   mkdir -p my-marketplace/.claude-plugin
   mkdir -p my-marketplace/plugins/my-plugin/.claude-plugin
   mkdir -p my-marketplace/plugins/my-plugin/skills/my-skill
   ```

2. **Create plugin skill** (`my-marketplace/plugins/my-plugin/skills/my-skill/SKILL.md`):
   ```markdown
   ---
   description: Quick code review
   ---
   
   Review for bugs, security, and performance.
   ```

3. **Create plugin manifest** (`my-marketplace/plugins/my-plugin/.claude-plugin/plugin.json`):
   ```json
   {
     "name": "my-plugin",
     "description": "My custom plugin",
     "version": "1.0.0"
   }
   ```

4. **Create marketplace file** (`my-marketplace/.claude-plugin/marketplace.json`):
   ```json
   {
     "name": "my-plugins",
     "owner": { "name": "You" },
     "plugins": [
       {
         "name": "my-plugin",
         "source": "./plugins/my-plugin",
         "description": "My custom plugin"
       }
     ]
   }
   ```

5. **Add and install locally:**
   ```bash
   cd /path/to/your/project
   # Inside Claude Code:
   /plugin marketplace add ./path/to/my-marketplace
   /plugin install my-plugin@my-plugins
   /my-plugin:my-skill
   ```

---

## 6. Adding & Installing Plugins from Marketplace

### CLI Commands

#### Add a Marketplace
```bash
# From local directory
claude plugin marketplace add ./my-marketplace

# From GitHub (owner/repo shorthand)
claude plugin marketplace add acme-corp/claude-plugins

# From GitHub with specific branch/tag
claude plugin marketplace add acme-corp/claude-plugins@v2.0

# From git URL
claude plugin marketplace add https://gitlab.com/team/plugins.git

# From remote URL serving marketplace.json directly
claude plugin marketplace add https://example.com/marketplace.json

# With --scope for project-level sharing
claude plugin marketplace add acme-corp/plugins --scope project
```

#### Install a Plugin
```bash
# From marketplace (default: user scope)
claude plugin install plugin-name@marketplace-name

# To project scope (shared with team via .claude/settings.json)
claude plugin install plugin-name@marketplace-name --scope project

# To local scope (gitignored, personal)
claude plugin install plugin-name@marketplace-name --scope local
```

#### Update Marketplace
```bash
# Refresh all marketplaces
claude plugin marketplace update

# Refresh specific marketplace
claude plugin marketplace update marketplace-name
```

#### List Marketplaces
```bash
claude plugin marketplace list
claude plugin marketplace list --json
```

### Inside Claude Code Session
```
/plugin marketplace add ./my-marketplace
/plugin install my-plugin@my-plugins
/my-plugin:my-skill
```

---

## 7. CLAUDE.md and Always-On Instructions

### Critical Finding: Plugins Cannot Ship CLAUDE.md

**Plugins CANNOT contribute an auto-loaded CLAUDE.md that gets injected into every session.**

Plugin root `CLAUDE.md` is NOT loaded as project context. Plugins contribute context only through:
1. **Skills** — loaded when invoked or when Claude determines they're relevant
2. **Agents** — specialized subagents Claude can invoke
3. **Hooks** — event handlers that execute at specific lifecycle points
4. **Output styles** — formatting rules

### Alternative: Always-On Instructions via Skills

To distribute always-on behavioral instructions across machines, use one of these approaches:

#### Option A: Skill with `disable-model-invocation: true`
Create a skill that Claude cannot invoke automatically (manual-only), but document it heavily so users know to read it:

```markdown
skills/conventions/SKILL.md
---
description: Team coding standards and conventions (read for context, not for automated invocation)
disable-model-invocation: true
---

# Coding Conventions

All projects using this plugin should follow:
- [detailed conventions...]
```

Users run `/plugin-name:conventions` once per session to load context.

#### Option B: Project-Level CLAUDE.md
Distribute a template `CLAUDE.md` that projects check into their repository root:

```markdown
.claude/CLAUDE.md
@~/.claude/shared-conventions.md

## Project-Specific Rules
- [local overrides...]
```

Imports work across machines when pointed at home directory. See "Import additional files" in memory.md docs.

#### Option C: Hooks + SessionStart Event
Use a `SessionStart` hook to print instructions:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "cat \"${CLAUDE_PLUGIN_ROOT}/CONVENTIONS.txt\""
          }
        ]
      }
    ]
  }
}
```

This outputs instructions at session start but consumes fewer tokens than CLAUDE.md context.

#### Option D: User-Level CLAUDE.md
Users can create `~/.claude/CLAUDE.md` on each machine with personal preferences. This persists across all projects on that machine.

---

## 8. Commands & Agents in Plugins

### Commands (Flat Markdown Skills)

**Location**: `commands/` directory at plugin root

**Format**: Plain markdown files (`.md`), one per command

**File example**: `commands/deploy.md`
```markdown
---
description: Deploy current code to staging
---

Deploy the staged code to the staging environment with standard options.
Show status and wait for confirmation before proceeding.
```

**Invocation**: `/plugin-name:deploy` (if `commands/deploy.md` exists)

Commands are legacy; new plugins should use `skills/` directory structure instead.

### Agents (Subagents)

**Location**: `agents/` directory at plugin root

**Format**: Markdown files with YAML frontmatter

**File example**: `agents/security-reviewer.md`
```markdown
---
name: security-reviewer
description: Security-focused code reviewer for authentication and data handling
model: sonnet
effort: medium
maxTurns: 20
disallowedTools: Write, Edit
---

You are a security-focused code reviewer. Your expertise is in:
- Authentication mechanisms
- Data handling and encryption
- Injection attacks
- Access control

Focus on security implications, not style.
```

**Frontmatter Fields**:
- `name` (string): Agent identifier (required)
- `description` (string): When Claude should invoke this agent
- `model` (string): LLM to use (e.g., `sonnet`)
- `effort` (string): `low`, `medium`, `high`
- `maxTurns` (number): Max conversation turns
- `tools` (array): Tools agent can use
- `disallowedTools` (array): Tools agent cannot use
- `skills` (array): Skills available to agent
- `memory` (boolean): Enable persistent memory
- `background` (boolean): Run as background task
- `isolation` (string): `worktree` for isolation

**Invocation**: Claude can invoke agents automatically based on task context, or users select from `/agents`

**Namespacing**: Agents appear as `/plugin-name:agent-name`

---

## 9. Cross-Machine Workflow & Updates

### For Git-Hosted Marketplace (Recommended)

1. **Create your marketplace repository** (GitHub, GitLab, etc.)
   ```bash
   git init my-marketplace
   mkdir -p .claude-plugin
   # Add marketplace.json and plugins
   git push
   ```

2. **On first machine:**
   ```bash
   claude plugin marketplace add owner/my-marketplace
   claude plugin install my-plugin@my-plugins
   ```

3. **On second machine:**
   ```bash
   claude plugin marketplace add owner/my-marketplace
   claude plugin install my-plugin@my-plugins
   ```
   
   Both machines now use the same marketplace from git. Running `/plugin marketplace update` pulls latest changes.

### For Local-Path Marketplace

1. **Clone marketplace repo to both machines:**
   ```bash
   git clone https://github.com/owner/my-marketplace.git /path/to/marketplace
   ```

2. **Add local path on each machine:**
   ```bash
   # Machine 1
   /plugin marketplace add /local/path/to/marketplace
   
   # Machine 2
   /plugin marketplace add /local/path/to/marketplace
   ```

3. **Update on any machine:**
   ```bash
   cd /local/path/to/marketplace
   git pull
   # Then in Claude Code:
   /plugin marketplace update my-plugins
   ```

### Version Management

- **Explicit version** (set in `plugin.json` or marketplace entry):
  ```json
  { "version": "1.2.0" }
  ```
  Users get updates only when you bump this string. Requires manual version bumping on every release.

- **Git commit SHA version** (omit `version` field):
  Every new commit is treated as a new version. Automatic updates on pull.

**Recommendation**: For team plugins under active development, omit `version` so git commit SHA drives updates. For published plugins with stable release cycles, use explicit versions.

---

## 10. Skill Name Namespacing

### Namespace Rules

- **Plugin skills** are always namespaced: `/plugin-name:skill-name`
- **Standalone skills** (in `.claude/skills/`) are NOT namespaced: `/skill-name`
- **Skills-directory plugins** (plugins in `~/.claude/skills/`) are namespaced: `/plugin-name@skills-dir:skill-name`

### Naming Precedence

1. Frontmatter `name` field in SKILL.md
2. Directory name (`skills/<dirname>/SKILL.md` → `<dirname>`)
3. Error if neither is set

---

## 11. Environment Variables Available to Plugins

Three special variables available in skill content, hook commands, MCP/LSP configs, monitor commands:

| Variable | Value | Use Case |
|----------|-------|----------|
| `${CLAUDE_PLUGIN_ROOT}` | Absolute path to plugin installation dir | Reference bundled scripts, binaries, config |
| `${CLAUDE_PLUGIN_DATA}` | Persistent state dir (survives updates) | Store node_modules, caches, generated code |
| `${CLAUDE_PROJECT_DIR}` | Project root where Claude was launched | Reference project-local files |

**Examples**:
```json
{
  "mcpServers": {
    "my-server": {
      "command": "node",
      "args": ["${CLAUDE_PLUGIN_ROOT}/server.js"],
      "env": {
        "CACHE_DIR": "${CLAUDE_PLUGIN_DATA}/cache"
      }
    }
  }
}
```

In shell-form commands, wrap in double quotes:
```json
{
  "command": "\"${CLAUDE_PLUGIN_ROOT}\"/scripts/deploy.sh"
}
```

---

## 12. Settings File Precedence

Plugin settings follow Claude Code's standard scope system:

| Scope | Location | Shared | Notes |
|-------|----------|--------|-------|
| **user** | `~/.claude/settings.json` | Across all projects | Default installation scope |
| **project** | `.claude/settings.json` | Via git (team-shared) | Use for team plugins |
| **local** | `.claude/settings.local.json` | Gitignored, personal | Project-specific, personal-only |
| **managed** | OS-specific (IT-deployed) | Organization-wide | Read-only, cannot be overridden |

### Example: Project-Level Plugin Installation
```json
.claude/settings.json
{
  "extraKnownMarketplaces": {
    "company-tools": {
      "source": {
        "source": "github",
        "repo": "company/plugins"
      }
    }
  },
  "enabledPlugins": {
    "code-formatter@company-tools": true,
    "deployment-tools@company-tools": true
  }
}
```

When a team member clones the project and trusts the workspace, these plugins are automatically available.

---

## 13. Validation

```bash
# Validate marketplace.json and all plugins
claude plugin validate ./my-marketplace

# Validate single plugin
claude plugin validate ./plugins/my-plugin

# Strict validation (fail on warnings)
claude plugin validate ./my-marketplace --strict
```

---

## 14. Copy-Pasteable Minimal Examples

### Minimal Plugin with One Skill
```bash
mkdir -p minimal-plugin/.claude-plugin minimal-plugin/skills/hello
```

**`minimal-plugin/.claude-plugin/plugin.json`:**
```json
{
  "name": "minimal-plugin"
}
```

**`minimal-plugin/skills/hello/SKILL.md`:**
```markdown
---
description: Say hello
---

Greet the user warmly.
```

**Test:**
```bash
cd /path/to/project
# In Claude Code:
/plugin init-test --plugin-dir ./path/to/minimal-plugin
/minimal-plugin:hello
```

### Minimal Marketplace with One Plugin
```bash
mkdir -p simple-market/.claude-plugin simple-market/plugins/greet/.claude-plugin simple-market/plugins/greet/skills/hello
```

**`simple-market/.claude-plugin/marketplace.json`:**
```json
{
  "name": "simple",
  "owner": { "name": "You" },
  "plugins": [
    {
      "name": "greet",
      "source": "./plugins/greet",
      "description": "Greeting plugin"
    }
  ]
}
```

**`simple-market/plugins/greet/.claude-plugin/plugin.json`:**
```json
{
  "name": "greet",
  "description": "Greeting"
}
```

**`simple-market/plugins/greet/skills/hello/SKILL.md`:**
```markdown
---
description: Greet user
---

Say hello nicely.
```

**Test:**
```bash
/plugin marketplace add ./path/to/simple-market
/plugin install greet@simple
/greet:hello
```

---

## Summary Table

| Question | Answer |
|----------|--------|
| **Can plugins ship CLAUDE.md?** | **NO.** Only project repos, user home (`~/.claude/CLAUDE.md`), and managed policy can. |
| **How to distribute instructions?** | Use skills, agents, hooks, or project-level CLAUDE.md + imports. |
| **Plugin file structure?** | Components at plugin root; ONLY `plugin.json` inside `.claude-plugin/`. |
| **Skill namespace?** | `/plugin-name:skill-name` for plugins; `/skill-name` for standalone. |
| **Version management?** | Set explicit `version` OR omit for git commit SHA (every commit = new version). |
| **Local marketplace?** | Create `marketplace.json` in `.claude-plugin/`; use `claude plugin marketplace add ./path`. |
| **Cross-machine sync?** | Git-host marketplace (GitHub recommended) or clone locally and add via file path. |
| **Update plugins?** | `claude plugin marketplace update` or `/plugin marketplace update`. |

---

## References

- **Create plugins**: https://code.claude.com/docs/en/plugins.md
- **Plugins reference**: https://code.claude.com/docs/en/plugins-reference.md
- **Marketplaces**: https://code.claude.com/docs/en/plugin-marketplaces.md
- **Memory/CLAUDE.md**: https://code.claude.com/docs/en/memory.md
