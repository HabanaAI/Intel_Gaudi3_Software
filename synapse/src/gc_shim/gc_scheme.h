#pragma once

static const char s_graphCompilerScheme[] =
    R"(
    {
        "internal": true,
        "about": {
            "version": "0.1.0",
            "description": [
                "- Record pre compiled Synapse graphs to file.",
                "- Record launch tensors data to file."
            ]
        },
        "graph" : {
            "guiSequence": 0,
            "displayName": "Graph Record",
            "path": {
                "guiSequence": 0,
                "displayName": "File/Folder Path",
                "description": "Relative file/folder path for the recorded files",
                "type": "text",
                "defaultValue": ""
            },
            "split": {
                "guiSequence": 1,
                "displayName": "Split By Graphs",
                "description": "Write each synapse graph to a single file",
                "type": "checkbox",
                "defaultValue": false
            },
            "type": {
                "guiSequence": 2,
                "displayName": "Graph Type",
                "description": "Select graph types",
                "pre": {
                    "guiSequence": 0,
                    "displayName": "Pre",
                    "type": "checkbox",
                    "defaultValue": false
                },
                "post": {
                    "guiSequence": 1,
                    "displayName": "Post",
                    "type": "checkbox",
                    "defaultValue": false
                },
                "passes": {
                    "guiSequence": 2,
                    "displayName": "Passes",
                    "description": "Passes Graph Config",
                    "enable": {
                        "guiSequence": 0,
                        "displayName": "Enable",
                        "type": "checkbox",
                        "defaultValue": false
                    },
                    "pass": {
                        "guiSequence": 1,
                        "displayName": "Pass",
                        "type": "text",
                        "defaultValue": ""
                    }
                }
            }
        },
        "tensors" : {
            "guiSequence": 1,
            "displayName": "Data Record",
            "path": {
                "guiSequence": 0,
                "displayName": "File/Folder Path",
                "description": "Relative file/folder path for the recorded files",
                "type": "text",
                "defaultValue": ""
            },
            "split": {
                "guiSequence": 1,
                "displayName": "Split By Graphs",
                "description": "Write each synapse graph to a single file",
                "type": "checkbox",
                "defaultValue": false
            },
            "min_iter": {
                "guiSequence": 2,
                "displayName": "Begin Record Iteration",
                "description": "Select first iteration to record",
                "type": "text",
                "defaultValue": 0
            },
            "max_iter": {
                "guiSequence": 3,
                "displayName": "End Record Iteration",
                "description": "Select last iteration to record",
                "type": "text",
                "defaultValue": -1
            },
            "overwrite_last_iteration": {
                "guiSequence": 4,
                "displayName": "Overwrite last iteration",
                "description": "Record only last iteration",
                "type": "checkbox",
                "defaultValue": false
            },
            "compression": {
                "guiSequence": 5,
                "displayName": "Compression",
                "type": "select",
                "defaultValue": "1",
                "options": [
                    [0, "none"],
                    [1, "lz4"]
                ],
                "description": "Data compression type"
            }
        }
    }
)";