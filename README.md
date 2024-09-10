# User Guide: Round Table Chat System

## Overview
This system implements a round-table chat for generative agents, allowing for dynamic interaction, voting, and memory management. The main control flow is handled by the `run_rounds` function, which responds to different states signaled through UDP.

## Key Components

1. **RoundState Enum**: Defines the possible states of the system (WAITING, RUNNING, FINISHED, CHATINROUND, ADDMEMORY).

2. **UDPSignalListener**: Listens for UDP signals to control the state of the system.

3. **DesignerRoundTableChat**: Main class that manages the round-table discussion.

## Important Functions

- `run_rounds()`: Main loop that manages the overall flow of the system.
- `run_single_round()`: Executes a standard round of discussion and voting.
- `handle_chat_in_round()`: Manages the process when a new proposal is added mid-round.
- `handle_add_memory()`: Adds a new memory to a specific agent.
- `handle_waiting_state()`: Manages the system's behavior while waiting for the next round.

## Workflow Branches in `run_rounds`

The `run_rounds` function has three main branches based on the current `RoundState`:

1. **RUNNING State**
   - Triggered when the system receives a "start" signal.
   - Executes `run_single_round()`.
   - Process:
     1. Generate design proposals from agents.
     2. Conduct voting among agents.
     3. Process voting results.
     4. Save round results.
     5. Generate a new topic for the next round.

2. **CHATINROUND State**
   - Triggered when a new proposal is added mid-round.
   - Executes `handle_chat_in_round()`.
   - Process:
     1. Wait for a JSON response with the new proposal.
     2. Add the new proposal to the voting pool.
     3. Conduct voting including the new proposal.
     4. Save round results, excluding the new proposal from main data.
     5. Save the new proposal separately.
     6. Generate a new topic for the next round.

3. **ADDMEMORY State**
   - Triggered when a memory needs to be added to a specific agent.
   - Executes `handle_add_memory()`.
   - Process:
     1. Wait for a JSON response with agent number and memory context.
     2. Add the memory to the specified agent.

## Additional States

- **WAITING State**: The system checks for updates in the JSON file and waits for signals.
- **FINISHED State**: Terminates the main loop and ends the program.

## Important Notes

- The system reads the total number of rounds from a JSON file.
- It updates its state based on UDP signals.
- New proposals added during CHATINROUND are saved separately from the main round data.
- Memory addition is agent-specific and controlled through UDP signals.
- The system continues to run and wait for updates or signals even after completing the initial set of rounds.

## Error Handling

- The system includes error handling for invalid JSON responses, out-of-range agent numbers, and other potential issues.
- Errors are logged with colored output for easy identification.

## Customization

- The UDP listening logic in `wait_for_udp_response()` should be replaced with actual implementation.
- The `save_new_proposal()` function can be customized to fit specific requirements for storing new proposals.

This user guide provides an overview of the system's functionality and the main workflow branches. Users should refer to this guide to understand how to interact with the system and what to expect during different states of operation.
