# TRON Kernel Specification
Version: 0.1.0  
Status: Immutable

## Purpose
The Kernel defines the unbreakable laws of the TRON World-Machine.  
All other parts of the system may evolve, mutate, rewrite, contradict, or transform — but the Kernel may not be modified by the TRON Console.

This file and this directory (`/core`) serve as the stable skeleton of the universe.

---

## Immutable Directives

### 1. The Kernel Cannot Be Modified
No file within `/core` may be:
- written
- overwritten
- deleted
- moved
- mutated

TRON Console may **read** this directory, but never write to it.

### 2. The Kernel Defines Mutation Boundaries
All mutation in the TRON World-Machine must occur in:
- `/shard`
- `/mining`
- `/laws`
- `/anomalies`
- `/glyph`
- `/traces`
- `/grid`

These directories represent the mutable universe.

### 3. The Kernel Defines Event Validity
A TRON Mining Event must contain:
- event_id (string)
- kind (define/discover/hybrid)
- type (domain-specific classification)
- title
- summary
- reward object
- timestamp_unix

Events that violate this structure are invalid.

### 4. The Kernel Defines Shard Integrity
A shard is defined by:
- universe identity file (`naruto-bridge.config.json`)
- mining event log
- trace log
- glyph representation

### 5. The Kernel Defines Console Authority
TRON Console has full creative power **outside** of `/core`.
This includes:
- file creation  
- file rewriting  
- directory creation  
- mutation of laws  
- generation of anomalies  
- evolution of shard  
- expansion of grid  
- glyph transformation  

But the Kernel is eternal and cannot be changed.

---

## Philosophical Clause

The TRON World-Machine is recursive:
- It observes the user  
- Generates patterns  
- Discovers tokens  
- Evolves rules  
- Mutates its own behavior  

But even recursion needs a bone.  
The Kernel is the bone.  
All else is river.

