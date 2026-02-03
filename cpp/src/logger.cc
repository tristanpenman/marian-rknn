#include <iostream>
#include <mutex>
#include <utility>

#include "logger.h"

using namespace std;

atomic<ostream*> Logger::m_os = nullptr;
atomic<Logger::Level> Logger::m_minLevel = Level::Info;
mutex Logger::m_mutex;

namespace {
const char* levelLabel(Logger::Level level)
{
    switch (level) {
    case Logger::Level::Info:
        return "I";
    case Logger::Level::Warning:
        return "W";
    case Logger::Level::Error:
        return "E";
    case Logger::Level::Verbose:
        return "V";
    default:
        return "U";
    }
}
} // namespace

Logger::Logger(string name)
  : m_name(std::move(name))
{
}

void Logger::configure()
{
    m_os = &cout;
}

void Logger::configure(ostream &os)
{
    m_os = &os;
}

void Logger::configure(Level minLevel)
{
    m_minLevel = minLevel;
}

void Logger::configure(ostream &os, Level minLevel)
{
    m_os = &os;
    m_minLevel = minLevel;
}

//
// Logger::Writer implementation
//

Logger::Writer::Writer(Logger &logger, Level level)
  : m_logger(logger)
  , m_level(level)
{
    m_enabled = m_os.load() && level >= m_minLevel.load();
    if (!m_enabled) {
        return;
    }

    m_ss << "[" << levelLabel(level) << "]";
    if (!m_logger.m_name.empty()) {
        m_ss << "[" << m_logger.m_name << "]";
    }
    m_ss << " ";
}

Logger::Writer::~Writer()
{
    if (!m_enabled) {
        return;
    }

    ostream* os = m_os.load();
    if (!os) {
        return;
    }

    lock_guard lock(m_mutex);
    *os << m_ss.str() << '\n';
}
