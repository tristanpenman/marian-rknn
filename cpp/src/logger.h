#pragma once

#include <atomic>
#include <ostream>
#include <sstream>
#include <string>
#include <mutex>

/**
 * Simple streaming logger
 *
 * This logger class is designed for ease-of-use and convenience, while providing performance and thread safety.
 *
 * It is intended to be used like this:
 *
 *   #include "Logger.h"
 *
 *   #undef LOG
 *   #define LOG Logger("MyCategory")
 *
 *   void myFunction()
 *   {
 *     // enable logging, uses std::cout by default
 *     Logger::configure();
 *
 *     // log info by default
 *     LOG() << "Log some stuff, maybe even some value in hex: 0x" << hex << 23030;
 *     LOG(WARNING) << "This is a warning message";
 *   }
 *
 * This would write the following to std::cout:
 *
 *   [I][MyCategory] Log some stuff, maybe even some value in hex: 0x59F6
 *   [W][MyCategory] This is a warning message
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
        Verbose = 0,
        Info = 1,
        Warning = 2,
        Error = 3
    };

    class Writer
    {
    public:
        Writer(Logger &logger, Level level);
        ~Writer();

        template<typename T>
        Writer& operator<<(T const & value)
        {
            if (m_enabled) {
                m_ss << value;
            }

            return *this;
        }

    private:
        Logger &m_logger;
        Level m_level;
        std::stringstream m_ss;
        bool m_enabled{false};
    };

    explicit Logger(std::string name = {});

    Writer operator()(const Level level = Level::Info)
    {
        return Writer{*this, level};
    }

    static void configure();
    static void configure(std::ostream& os);
    static void configure(Level minLevel);
    static void configure(std::ostream& os, Level minLevel);

    static bool verbose()
    {
        return m_minLevel.load() <= Level::Verbose;
    }

private:
    static std::atomic<std::ostream*> m_os;
    static std::atomic<Level> m_minLevel;
    static std::mutex m_mutex;

    std::string m_name;
};

#define LOG     Logger()
#define INFO    Logger::Level::Info
#define WARNING Logger::Level::Warning
#define ERROR   Logger::Level::Error
#define VERBOSE Logger::Level::Verbose
