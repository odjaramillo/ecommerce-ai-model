# Skill Registry

**Delegator use only.** Any agent that launches sub-agents reads this registry to resolve compact rules, then injects them directly into sub-agent prompts. Sub-agents do NOT read this registry or individual SKILL.md files.

See `_shared/skill-resolver.md` for the full resolution protocol.

## User Skills

| Trigger | Skill | Path |
|---------|-------|------|
| When user says \"judgment day\", \"judgment-day\", \"review adversarial\", \"dual review\", \"doble review\", \"juzgar\", \"que lo juzguen\"." | judgment-day | /home/odjt/.config/opencode/skills/judgment-day/SKILL.md |
| When writing Go tests, using teatest, or adding test coverage." | go-testing | /home/odjt/.config/opencode/skills/go-testing/SKILL.md |
| When user asks to create a new skill, add agent instructions, or document patterns for AI." | skill-creator | /home/odjt/.config/opencode/skills/skill-creator/SKILL.md |
| When creating a pull request, opening a PR, or preparing changes for review." | branch-pr | /home/odjt/.config/opencode/skills/branch-pr/SKILL.md |
| When creating a GitHub issue, reporting a bug, or requesting a feature." | issue-creation | /home/odjt/.config/opencode/skills/issue-creation/SKILL.md |
| when a PR would exceed 400 changed lines, when planning chained PRs, stacked PRs, or reviewable slices." | chained-pr | /home/odjt/.config/opencode/skills/chained-pr/SKILL.md |
| when writing guides, READMEs, RFCs, onboarding docs, architecture docs, or review-facing documentation." | cognitive-doc-design | /home/odjt/.config/opencode/skills/cognitive-doc-design/SKILL.md |
| when drafting or posting feedback, review comments, maintainer replies, Slack messages, or GitHub comments." | comment-writer | /home/odjt/.config/opencode/skills/comment-writer/SKILL.md |
| when implementing a change, preparing commits, splitting PRs, or planning chained or stacked PRs." | work-unit-commits | /home/odjt/.config/opencode/skills/work-unit-commits/SKILL.md |
| None | find-skills | /home/odjt/.agents/skills/find-skills/SKILL.md |

## Compact Rules

Pre-digested rules per skill. Delegators copy matching blocks into sub-agent prompts as `## Project Standards (auto-resolved)`.

### judgment-day
- Follow standard practices for judgment-day

### go-testing
- Follow standard practices for go-testing

### skill-creator
- Follow standard practices for skill-creator

### branch-pr
- Follow standard practices for branch-pr

### issue-creation
- Follow standard practices for issue-creation

### chained-pr
- Follow standard practices for chained-pr

### cognitive-doc-design
- Follow standard practices for cognitive-doc-design

### comment-writer
- Follow standard practices for comment-writer

### work-unit-commits
- Follow standard practices for work-unit-commits

### find-skills
- Follow standard practices for find-skills

## Project Conventions

| File | Path | Notes |
|------|------|-------|
