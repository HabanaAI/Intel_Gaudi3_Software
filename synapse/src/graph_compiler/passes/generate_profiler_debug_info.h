#pragma once

class HabanaGraph;

// Perform a relatively heavy init required for the pass which is done once per process.
void initializeProfilerDebugInfo();

/*
 * Each execution engine type maintains its own execution order based on the final execution order.
 * The context id field in the first node in each execution engine type contains the graph id.
 */
bool generateProfilerDebugInfo(HabanaGraph& g);