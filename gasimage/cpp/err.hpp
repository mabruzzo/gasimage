#include <cstdarg> // va_list and friends
#include <cstdio>
#include <cstdlib> // abort
#include <string>

/// helper function that prints an error message & aborts the program (in
/// Commonly invoked through a macro.
[[noreturn]] inline void Abort_With_Err_(const char* func_name,
                                         const char* file_name,
                                         int line_num, const char* msg, ...)
{
  // to handle cases where all processes encounter the same error, we
  // pre-buffer the error message (so that the output remains legible)

  // since we are aborting, it's OK that this isn't the most optimized

  // prepare some info for the error message header
  const char* sanitized_func_name = (func_name == nullptr) ?
    "{unspecified}" : func_name;

  // prepare the formatted message
  std::string msg_buf;
  if (msg == nullptr) {
    msg_buf = "{nullptr encountered instead of error message}";
  } else {
    std::va_list args, args_copy;
    va_start(args, msg);
    va_copy(args_copy, args);

    std::size_t bufsize_without_terminator = std::vsnprintf(nullptr, 0, msg,
                                                            args);
    va_end(args);

    // NOTE: starting in C++17 it's possible to mutate msg_buf by mutating
    // msg_buf.data()

    // we initialize a msg_buf with size == bufsize_without_terminator (filled
    // with ' ' chars)
    // - msg_buf.data() returns a ptr with msg_buf.size() + 1 characters. We
    //   are allowed to mutate any of the first msg_buf.size() characters. The
    //   entry at msg_buf.data()[msg_buf.size()] is initially  '\0' (& it MUST
    //   remain equal to '\0')
    // - the 2nd argument of std::vsnprintf is the size of the output buffer.
    //   We NEED to include the terminator character in this argument,
    //   otherwise the formatted message will be truncated
    msg_buf = std::string(bufsize_without_terminator, ' ');
    std::vsnprintf(msg_buf.data(), bufsize_without_terminator + 1, msg,
                   args_copy);
    va_end(args_copy);
  }

  // now write the error and exit
  std::fprintf(stderr,
               "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n"
               "Error occurred in %s on line %d\n"
               "Function: %s\n"
               "Message: %s\n"
               "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n",
               file_name, line_num, sanitized_func_name, msg_buf.data());
  std::fflush(stderr);  // may be unnecessary for stderr
  std::abort();
}

/// __MY_PRETTY_FUNC__ is a magic constant like __LINE__ or __FILE__ that
/// provides the name of the current function.
/// - The C++11 standard requires that __func__ is provided on all platforms,
///   but that only provides limited information (just the name of the
///   function).
/// - Where available, we prefer to use compiler-specific features that provide
///   more information about the function (like the scope of the function & the
///   the function signature).
#ifdef __GNUG__
  #define __MY_PRETTY_FUNC__ __PRETTY_FUNCTION__
#else
  #define __MY_PRETTY_FUNC__ __func__
#endif

/// @brief print an error-message (with printf formatting) & abort the program.
///
/// This macro should be treated as a function with the signature:
///   [[noreturn]] void ERROR(const char* msg, ...);
///
/// - The 1st arg is printf-style format argument specifying the error message
/// - The remaining args arguments are used to format error message
///
/// @note
/// the ``msg`` string is part of the variadic args so that there is always
/// at least 1 variadic argument (even in cases when ``msg`` doesn't format
/// any arguments). There is no way around this until C++ 20.
#define ERROR(...) Abort_With_Err_(__MY_PRETTY_FUNC__, __FILE__, __LINE__, __VA_ARGS__)
