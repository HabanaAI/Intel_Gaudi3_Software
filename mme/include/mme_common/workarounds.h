#ifndef MME__WORKAROUNDS_H
#define MME__WORKAROUNDS_H

// Readme
//
// This file is the dashboard of all workarounds for all projects.
// (1) In case you have a "TO-DO", a workaround-related logic or any kind of temporary behavior, please define a new
//     macro in this file to be easy (or even possible) to follow up.
// (2) In case you cleaned the logic described at (1) and there is no need for keeping the macro, please remove it
//     from this file to keep reflecting an up-to-date status.
// (3) In case you were reviewing the code and were aware of a logic that require such macros, please go ahead and
//     add a macro for it.
// (4) All active macros in this file require further code development, each one tells a story on why it exists. It's
//     recommended to add a ticket number for each one, so you can add there tests or extra details.
//

//
// Common Workarounds
//

// All temporary or workaround code must be activated using this macro definition:
// When there is no recipe workarounds, this macro can be set to 0 - indicating no workarounds.
#define ENABLE_RECIPE_WORKAROUNDS 1

//
// Gaudi 1 Workarounds
//

// Never read partial splits in revere order
#define GAUDI_DEBUG_DISABLE_REVERSED_PARTIALS 1

// Ignore SB reuse in recipe
#define GAUDI_DEBUG_IGNORE_SB_REUSE_IN_RECIPE 1

// Gaudi 2 Workarounds
//
// Power test undone work or mysterious Goya2 logic that needs to be evaluated for Gaudi2
#define POWER_TEST_TODO 0

#endif //MME__WORKAROUNDS_H
