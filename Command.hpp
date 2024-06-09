#ifndef COMMAND_H
#define COMMAND_H

#include <Arduino.h>

namespace model {

/**
 * @brief シリアル通信用コマンドモデル
 */
class Command {
   public:
    Command(String command, double value);

    Command(String serialStr);

    /**
     * @brief コマンド文字列をパース
     * 
     * @param serialStr 
     */
    void parseCommand(String serialStr);

    /**
     * @brief Get the Command string of Command
     * 
     * @return String 
     */
    String getCommand() const;

    /**
     * @brief Get the Value of Command
     * 
     * @return double 
     */
    double getValue() const;

   private:
    String command_;
    double value_;
};
}  // namespace model

#endif
