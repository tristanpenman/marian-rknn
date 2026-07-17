# Conventions

These conventions apply to project-owned code. Third-party code should not be reformatted merely to match this document. At an integration boundary, retain the spelling and idioms of the external API, but use the project conventions inside project-owned wrappers and implementations.

When modifying older code, apply these conventions to the code being changed when doing so is low risk. Avoid unrelated, file-wide formatting or renaming changes.

## C++

### Files and formatting

- Use `.h` for headers and the source extension already established by the project (`.cc` or `.cpp`) for implementations.
- Indent with four spaces. Do not use tabs.
- Put opening braces on the next line for functions, classes, structs, unions, and enums.
- Put opening braces on the same line for namespaces and control-flow statements.
- Always use braces for the body of `if`, `else`, `for`, `while`, and similar statements, including single-statement bodies.
- Put a space on both sides of the colon in a range-based `for` loop.
- Bind pointer and reference markers to the type: `Widget* widget`, `const Vector& value`.
- Keep short getters and setters in the normal multi-line form. An inline one-line definition is acceptable when it materially improves the readability of a long, repetitive class.
- For a multi-line function declaration or definition, indent each parameter by four spaces and keep the closing parenthesis with the opening brace.
- Put each member on its own line in a multi-member constructor initializer list.
- Follow the established line-ending and maximum-line-length settings of the repository.

```cpp
Result calculateResult(
    const Input& input,
    std::size_t sampleCount)
{
    if (sampleCount == 0) {
        return {};
    }

    for (const auto& sample : input.samples()) {
        // Process the sample.
    }

    return makeResult(input, sampleCount);
}
```

### Names

- Classes, structs, unions, type aliases, and enum types use `PascalCase`.
- Functions and methods use `camelCase`.
- Local variables, parameters, and public struct fields use `camelCase`.
- Non-static data members use `camelCase` with a trailing underscore: `byteCount_`.
- Mutable global and namespace-scope variables use `camelCase` with a `g_` prefix: `g_shutdownRequested`. This includes variables in anonymous namespaces and `thread_local` variables because their storage duration is not apparent at the point of use. Prefer avoiding mutable global state when ownership or dependency injection is practical.
- Constants, `constexpr` values, and enumerators use `kPascalCase`: `kBufferSize`, `kInvalidArgument`.
- Preprocessor macros use `ALL_CAPS`. Prefer typed constants and functions to macros.
- Template parameters use `PascalCase`: `ValueType` or `T`.
- CUDA kernels use `camelCase` with a `Kernel` suffix.

Do not rename symbols owned by an external library or operating-system API. Where a project must implement a prescribed callback or interface, retain its required name. Project wrappers around such APIs use the normal project naming rules.

Legacy modules that consistently use `snake_case`, `m_` member prefixes, unprefixed PascalCase enumerators, or `ALL_CAPS` constants may retain that style until they are deliberately migrated. Do not mix competing styles within one class or closely related module.

### Headers and includes

- Use `#pragma once` in project headers.
- Include what the file directly uses.
- Group includes in this order, with one blank line between groups:
  1. The corresponding header, in an implementation file.
  2. C and C++ standard-library headers.
  3. Third-party and platform headers.
  4. Project headers.
- Sort includes consistently within each group.
- Prefer a forward declaration when it is valid and makes the dependency clearer. Include the owning header when a complete type or a library-provided smart-pointer declaration is required.
- Keep platform-specific includes out of portable public headers where practical.

### Namespaces

- Use an anonymous namespace for implementation details confined to one translation unit.
- Add a namespace name to the closing-brace comment. Use `// namespace` for an anonymous namespace.
- Do not use `using namespace` in a header. Use it sparingly in a narrow implementation scope.
- Do not indent the contents of a namespace.
- Include an empty line after opening a namespace and before closing one.

```cpp
namespace {

int helper()
{
    return 1;
}

}  // namespace

namespace example {

// ...

}  // namespace example
```

### Types, ownership, and interfaces

- Prefer fixed-width integer types such as `std::int32_t` where width is part of the interface or data format. Otherwise use the natural type for the operation, such as `std::size_t` for sizes.
- Pass small, inexpensive value types by value. Pass larger read-only objects by `const&` and mutable non-owning objects by reference or pointer as appropriate.
- Use `std::unique_ptr` for exclusive ownership and `std::shared_ptr` only for genuine shared ownership.
- Pass smart pointers by value when transferring or sharing ownership. Do not use a smart pointer merely to express non-owning access.
- Prefer constructor injection when an object requires a dependency for its valid lifetime.
- Mark read-only methods `const` and use `const` for values that do not change.
- Use `auto` when the type is obvious from the initializer or when spelling the type would obscure the intent. Avoid it when the concrete type conveys important ownership, precision, or conversion information.
- Initialise members at their declarations when the same default applies to every constructor.
- Use scoped enums (`enum class`) unless implicit conversion or compatibility with an existing interface is required.

### Comments and documentation

- Write comments that explain intent, constraints, or non-obvious behaviour rather than restating the code.
- Use sentence case and punctuation for prose comments.
- Avoid full stops for short annotations and non-prose comments.
- Documentation comments are appropriate for public types and APIs whose contract is not evident from the declaration.
- Section comments may be used to make a long file easier to navigate, but should not compensate for an overly large class or function.

### Example

```cpp
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "project/bar.h"

namespace example {

/** Processes values using the supplied service. */
class Foo
{
public:
    enum class Status
    {
        kReady,
        kFailed
    };

    explicit Foo(std::unique_ptr<Bar> bar);

    std::int32_t count() const
    {
        auto total = kStartValue;
        if (enabled_) {
            for (const auto value : values_) {
                total += value;
            }
        }

        return total;
    }

private:
    std::unique_ptr<Bar> bar_;
    bool enabled_ = true;
    std::vector<std::int32_t> values_;

    static constexpr std::int32_t kStartValue = 3;
};

}  // namespace example
```
