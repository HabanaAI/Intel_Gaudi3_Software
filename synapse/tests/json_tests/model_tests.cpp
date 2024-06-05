#include "model_tests.h"
#include "model_loader.h"
#include <stdexcept>

namespace json_tests
{
ModelTest::ModelTest(const ArgParser& args)
: BaseTest(),
  m_comObjFilePath(args.getValue<std::string>(an_com_file)),
  m_compile(args.getValueOrDefault(an_compile, false)),
  m_run(args.getValueOrDefault(an_run, false))
{
    // in some scenarios periodic flush causes issues.
    // so it's required to disable periodic flush in such cases
    synapse::LogManager::instance().enablePeriodicFlush(false);
}

void ModelTest::run()
{
    if (m_compile)
    {
        CompileModelLoader mt(m_comObjFilePath);
        if (!mt.run())
        {
            throw std::runtime_error("model test compilation failed, error: " + mt.getError());
        }
    }

    if (m_run)
    {
        RunModelLoader mt(m_comObjFilePath);
        if (!mt.run())
        {
            throw std::runtime_error("model test run failed, error: " + mt.getError());
        }
    }
}
}  // namespace json_tests