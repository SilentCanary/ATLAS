Based on the repository structure, I'll create a new feature for gitignore functionality. Here's the implementation:

📁 src\commands\gitignore.cpp
```cpp
#include "gitignore.h"
#include "../utils/hash.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

void GitIgnoreCommand::addPattern(const std::string& pattern) {
    std::ofstream gitignore(".gitignore", std::ios::app);
    if (gitignore.is_open()) {
        gitignore << pattern << "\n";
        gitignore.close();
        std::cout << "Added pattern to .gitignore: " << pattern << std::endl;
    } else {
        std::cerr << "Error: Could not open .gitignore file" << std::endl;
    }
}

void GitIgnoreCommand::listPatterns() {
    std::ifstream gitignore(".gitignore");
    if (gitignore.is_open()) {
        std::string line;
        std::cout << "Patterns in .gitignore:" << std::endl;
        while (getline(gitignore, line)) {
            std::cout << line << std::endl;
        }
        gitignore.close();
    } else {
        std::cerr << "Error: .gitignore file not found" << std::endl;
    }
}

void GitIgnoreCommand::removePattern(const std::string& pattern) {
    std::ifstream gitignoreIn(".gitignore");
    if (!gitignoreIn.is_open()) {
        std::cerr << "Error: .gitignore file not found" << std::endl;
        return;
    }

    std::vector<std::string> lines;
    std::string line;
    bool found = false;

    while (getline(gitignoreIn, line)) {
        if (line != pattern) {
            lines.push_back(line);
        } else {
            found = true;
        }
    }
    gitignoreIn.close();

    if (found) {
        std::ofstream gitignoreOut(".gitignore");
        for (const auto& l : lines) {
            gitignoreOut << l << "\n";
        }
        gitignoreOut.close();
        std::cout << "Removed pattern from .gitignore: " << pattern << std::endl;
    } else {
        std::cout << "Pattern not found in .gitignore: " << pattern << std::endl;
    }
}

void GitIgnoreCommand::checkIgnored(const std::string& path) {
    std::ifstream gitignore(".gitignore");
    if (!gitignore.is_open()) {
        std::cerr << "Error: .gitignore file not found" << std::endl;
        return;
    }

    std::string line;
    bool ignored = false;
    fs::path filepath(path);

    while (getline(gitignore, line)) {
        if (line.empty()) continue;

        // Simple pattern matching (can be enhanced)
        if (filepath.filename().string().find(line) != std::string::npos) {
            ignored = true;
            break;
        }
    }

    std::cout << "Path '" << path << "' is " << (ignored ? "" : "not ") << "ignored" << std::endl;
}
```

📁 src\commands\gitignore.h
```cpp
#ifndef GITIGNORE_H
#define GITIGNORE_H

#include <string>

class GitIgnoreCommand {
public:
    static void addPattern(const std::string& pattern);
    static void listPatterns();
    static void removePattern(const std::string& pattern);
    static void checkIgnored(const std::string& path);
};

#endif // GITIGNORE_H
```