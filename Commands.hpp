#include <stdlib.h>
// Commands.hpp
#ifndef COMMANDS_HPP
#define COMMANDS_HPP

#include <Arduino.h>

namespace model {

class Command {
   public:
    Command(String command, int value, int id)
        : command_(command), value_(value), id_(id) {}

    Command(String serialStr) { parseCommand(serialStr); }

    void parseCommand(String serialStr) {
        int firstColonIndex = serialStr.indexOf(':');
        if (firstColonIndex == -1) {
            command_ = serialStr;
            value_ = 0;
            id_ = -1;
        } else {
            id_ = atoi(serialStr.substring(0, firstColonIndex).c_str());
            int secondColonIndex = serialStr.indexOf(':', firstColonIndex + 1);
            if (secondColonIndex == -1) {
                command_ = serialStr.substring(firstColonIndex + 1);
                value_ = 0;
            } else {
                command_ = serialStr.substring(firstColonIndex + 1, secondColonIndex);
                value_ = atoi(serialStr.substring(secondColonIndex + 1).c_str());
            }
        }
    }

    String getCommand() const { return command_; }

    int getValue() const { return value_; }

    int getID() const {return id_; }

   private:
    String command_;
    int value_;
    int id_;
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
    static const String DETACH_MOTOR;
    static const String GET_DISTANCE;
    
    // 指定されたコマンドが無視リストに含まれているかを判定
    static bool is_ignore(const String &cmd)
    {
        return (cmd == STOP_TEMP || cmd == CHECK);
    }

    // 前回と同じコマンドで無視するかを判定
    static bool is_ignore_same_prev(const String &cmd, const String &prev)
    {
        // 無視しない連続コマンドを配列で定義
        const String allowed_repeats[] = {STOP, SPD_UP, SPD_DOWN, CHECK, DETACH_MOTOR, GET_DISTANCE};
        
        // 配列内に指定コマンドが存在するかチェック
        for (const String& allowed_cmd : allowed_repeats) {
            if (cmd == allowed_cmd) {
                return false;  // 許可された連続コマンド
            }
        }

        // 許可された連続コマンドでなければ、前回のコマンドと同じ場合に無視
        return cmd == prev;
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
const String Commands::DETACH_MOTOR = "DETACH";
const String Commands::GET_DISTANCE = "D";

#endif // COMMANDS_HPP
