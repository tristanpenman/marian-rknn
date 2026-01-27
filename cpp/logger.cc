#include <iostream>
#include <mutex>

#include "logger.h"

using namespace std;

atomic<ostream*> Logger::m_os = nullptr;
atomic<Logger::Level> Logger::m_minLevel = Logger::Level::Info;
mutex Logger::m_mutex;

namespace {
const char* levelLabel(Logger::Level level)
{
    switch (level) {
    case Logger::Level::Info:
        return "INFO";
    case Logger::Level::Warning:
        return "WARNING";
    case Logger::Level::Error:
        return "ERROR";
    case Logger::Level::Verbose:
        return "VERBOSE";
    default:
        return "UNKNOWN";
    }
}
} // namespace

Logger::Logger(const string& name, Level level)
    : m_level(level)
{
    ostream* os = m_os.load();
    m_enabled = os && level >= m_minLevel.load();
    if (!m_enabled) {
        return;
    }

    m_ss << "[" << levelLabel(level) << "][" << name << "] ";
}

Logger::~Logger()
{
    if (!m_enabled) {
        return;
    }

    ostream* os = m_os.load();
    if (!os) {
        return;
    }

    lock_guard<mutex> lock(m_mutex);
    *os << m_ss.str() << '\n';
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
