# Developer Handoff — Regional Federated Learning

## Project

**Federated Reinforcement Learning for Smart Traffic Lights Simulation Using Multimodal Data**

This project uses **SUMO + TraCI + PyTorch + DQN** to train intelligent traffic-signal controllers at individual intersections.

The current project already contains the working local traffic simulation and DQN/RL pipeline. The next task is to add the **Regional Federated Learning layer**.

---

## Important: Inspect the Existing Project First

The complete current project folder will be provided alongside this document.

**Do not assume any project structure, filenames, class names, APIs, or existing implementation details.**

Before making changes:

1. Inspect the entire project.
2. Identify how the current DQN model is defined.
3. Identify how training is performed.
5. Identify how multiple traffic intersections/nodes are currently represented
6. Identify the current training/evaluation flow.

---

# Current Architecture

At the moment, each traffic intersection can operate as an independent RL node:

```text
Intersection Node
       │
       ▼
  Local Traffic Data
       │
       ▼
      DQN
       │
       ▼
Traffic Signal Action
       │
       ▼
     SUMO
       │
       ▼
   Reward / State
       │
       └──► Local DQN Training
```

The eventual system is hierarchical:

```text
Intersection DQNs
       │
       ▼
Regional Aggregator
       │
       ▼
Regional Model
       │
       ▼
Multiple Regions
       │
       ▼
Global Aggregator
```

The **global aggregator is NOT the current task**, since the regional aggregator must be done first.

---

# Current Task: Regional Federated Learning

Implement the first level of federated learning:

> **Multiple intersection-level DQN agents within a region should collaboratively train through a Regional Federated Aggregator.**

The regional aggregator should combine model updates from participating intersection nodes and produce an aggregated regional model.

Conceptually:

```text
             ┌──────────────┐
             │ Intersection │
             │    Node 1    │
             │     DQN      │
             └──────┬───────┘
                    │
                    │ model update
                    ▼
             ┌──────────────┐
             │              │
             │   Regional   │
             │  Aggregator  │
             │              │
             └──────┬───────┘
                    ▲
                    │ model update
             ┌──────┴───────┐
             │ Intersection │
             │    Node 2    │
             │     DQN      │
             └──────────────┘
```

For N intersections:

```text
Node 1 ──┐
Node 2 ──┤
Node 3 ──┼──► Regional Aggregator ──► Aggregated Model
...      │
Node N ──┘
```

---

# Expected Federated Workflow

The implementation should support a federated training round similar to:

1. Initialize a common DQN model.
2. Distribute the model to participating intersection nodes.
3. Each node trains the DQN locally using its own traffic environment/data.
4. Each node sends its **model parameters/update**, not its raw traffic data, to the Regional Aggregator.
5. The Regional Aggregator aggregates the participating models.
6. The resulting regional model is distributed back to the nodes.
7. Nodes continue local training from the aggregated model.
8. Repeat for multiple federated rounds.


---

# Important Design Requirements

### 1. Preserve Local Learning

Federated learning should be added **around the existing DQN training**, not replace the DQN implementation.

Each intersection must still be capable of learning from its own local traffic environment.

### 2. Do Not Share Raw Traffic Data

The intended architecture is:

```text
Local Traffic Data
       │
       ▼
 Local DQN Training
       │
       ▼
 Model Parameters / Updates
       │
       ▼
Regional Aggregator
```

Raw vehicle trajectories, traffic observations, sensor data, etc. should remain local to the node.

### 3. Reuse Existing DQN

Do not create a second DQN implementation if one already exists.

Integrate the federation with the project's existing model/training code.

### 4. Keep the Architecture Extensible

The regional implementation should eventually allow:

```text
Region A ──► Regional Model A ──┐
Region B ──► Regional Model B ──┼──► Future Global Aggregator
Region C ──► Regional Model C ──┘
```

Do not implement the global aggregator yet, but avoid designing the regional component in a way that makes the future hierarchy difficult.

---

# What Should Be Demonstrated

After implementation, there should be a simple reproducible demonstration showing that:

- Multiple intersection nodes can train independently.
- Their DQN models can be sent to a regional aggregator.
- The aggregator can combine the models.
- The aggregated model can be returned to the nodes.
- Another local-training round can continue from the aggregated model.
- The existing SUMO/RL functionality still works.

Ideally, the implementation should make it possible to compare:

```text
Independent Local DQN
        vs
Regional Federated DQN
```

using the existing traffic-performance metrics.

---

# Expected Deliverables

Implement the regional federated learning functionality in the existing project.

The final implementation should include, as appropriate to the existing architecture:

- Regional aggregator
- Federated training-round logic
- Model parameter aggregation
- Distribution of aggregated parameters back to nodes
- Integration with existing DQN training
- A minimal runnable/demo federated training flow
- Clear comments/docstrings where the new logic is non-obvious

---

# Evaluation

Use the existing evaluation pipeline where possible.

The main purpose of this stage is to establish that regional collaboration works technically and does not break the existing RL system.

---


Move the project from:

```text
Multiple Independent DQN Intersection Nodes
```

to:

```text
Multiple DQN Intersection Nodes
            │
            ▼
   Regional Federated
      Aggregator
            │
            ▼
    Shared Regional
         Model
```

while preserving the existing SUMO + TraCI + DQN functionality.
