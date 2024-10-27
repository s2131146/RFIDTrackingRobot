// Commands.hpp
#ifndef COMMANDS_HPP
#define COMMANDS_HPP

#include <Arduino.h>

namespace model {

class Command {
   public:
    Command(String command, int value)
        : command_(command), value_(value) {}

    Command(String serialStr) { parseCommand(serialStr); }

    void parseCommand(String serialStr) {
        int colonIndex = serialStr.indexOf(':');
        if (colonIndex == -1) {
            command_ = serialStr;
            value_ = 0;
        } else {
            command_ = serialStr.substring(0, colonIndex);
            value_ = atof(serialStr.substring(colonIndex + 1).c_str());
        }
    }

    String getCommand() const { return command_; }

    int getValue() const { return value_; }

   private:
    String command_;
    int value_;
};
}  // namespace model

class Commands
{
public:
    // コマンド定数の宣言
    static const String STOP;
    static const String START;
    static const String GO_LEFT;
    static const String GO_RIGHT;
    static const String GO_CENTER;
    static const String GO_BACK;
    static const String SPD_UP;
    static const String SPD_DOWN;
    static const String SPD_RESET;
    static const String CHECK;
    static const String STOP_TEMP;
    static const String L_SPEED;
    static const String R_SPEED;
    static const String SET_DEFAULT_SPEED;
    static const String ROTATE_RIGHT;
    static const String ROTATE_LEFT;
    
    // 指定されたコマンドが無視リストに含まれているかを判定
    static bool is_ignore(const String &cmd)
    {
        return (cmd == STOP_TEMP || cmd == CHECK);
    }

    // 前回と同じコマンドで無視するかを判定
    static bool is_ignore_same_prev(const String &cmd, const String &prev)
    {
        // STOP, SPD_UP, SPD_DOWN 以外で前回と同じコマンドは無視
        return (cmd != STOP && cmd != SPD_UP && cmd != SPD_DOWN) && (prev == cmd);
    }
};

// 定数の初期化
const String Commands::STOP = "STOP";
const String Commands::START = "START";
const String Commands::GO_LEFT = "L";
const String Commands::GO_RIGHT = "R";
const String Commands::GO_CENTER = "C";
const String Commands::GO_BACK = "B";
const String Commands::SPD_UP = "UP";
const String Commands::SPD_DOWN = "DOWN";
const String Commands::SPD_RESET = "RESET";
const String Commands::CHECK = "YO";
const String Commands::STOP_TEMP = "STMP";
const String Commands::L_SPEED = "LS";
const String Commands::R_SPEED = "RS";
const String Commands::SET_DEFAULT_SPEED = "SD";
const String Commands::ROTATE_RIGHT = "RR";
const String Commands::ROTATE_LEFT = "RL";

#endif // COMMANDS_HPP
