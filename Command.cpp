#include "Command.hpp"

namespace model {

Command::Command(String command, double value)
    : command_(command), value_(value) {}

Command::Command(String serialStr) { parseCommand(serialStr); }

void Command::parseCommand(String serialStr) {
    int colonIndex = serialStr.indexOf(':');
    if (colonIndex == -1) {
        command_ = serialStr;
        value_ = 0.0;
    } else {
        command_ = serialStr.substring(0, colonIndex);
        value_ = atof(serialStr.substring(colonIndex + 1).c_str());
    }
}

String Command::getCommand() const { return command_; }

double Command::getValue() const { return value_; }
}  // namespace model
