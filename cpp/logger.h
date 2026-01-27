#pragma once

#include <atomic>
#include <ostream>
#include <sstream>
#include <string>
#include <mutex>

/**
 * Simple streaming logger
 *
 * This logger class is designed for ease-of-use and convenience, rather than for performance or thread safety.
 *
 * It is intended to be used like this:
 *
 *   #include "Logger.h"
 *
 *   #define LOG Logger("MyCategory")
 *
 *   void myFunction()
 *   {
 *     // enable logging, uses std::cout by default
 *     Logger::configure();
 *
 *     LOG << "Log some stuff, maybe even some value in hex: 0x" << hex << 23030;
 *   }
 *
 * This would write the following to std::cout:
 *
 *   [INFO][MyCategory] Log some stuff, maybe even some value in hex: 0x59F6
 *
 * The logger takes care of writing a new line to the end of the output, when its destructor is called. Content is
 * only written to the std::stringstream if the logger is configured and the log level is enabled.
 *
 * Although including <sstream> in this header is not great for compile times, this ensures that the necessary
 * stream operators are available for common types. And the compile time overhead could be mitigated using
 * precompiled headers.
 */
class Logger
{
public:
    enum class Level {
        Info = 0,
        Warning = 1,
        Error = 2,
        Verbose = 3
    };

    Logger(const std::string& name, Level level = Level::Info);

    ~Logger();

    template<typename T>
    Logger& operator<<(T const & value)
    {
        if (m_enabled) {
            m_ss << value;
        }

        return *this;
    }

    static void configure();
    static void configure(std::ostream& os);
    static void configure(Level minLevel);
    static void configure(std::ostream& os, Level minLevel);

private:
    static std::atomic<std::ostream*> m_os;
    static std::atomic<Level> m_minLevel;
    static std::mutex m_mutex;

    Level m_level;
    bool m_enabled{false};
    std::stringstream m_ss;
};
