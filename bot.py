import os
import logging
from graph import app
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from langchain_core.messages import HumanMessage, AIMessage


load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

chat_histories = {}
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES"))


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    await update.message.reply_html(
        f"Hi {user.mention_html()}! I'm your company's information assistant.\n\n"
        "I can remember the last few messages in our conversation. "
        "Ask me anything about our policies, projects, or history. For example:\n"
        "- What is our remote work policy?\n"
        "- Tell me about Project Example.\n"
        "- Who is the CEO of the Company?"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "I can answer questions based on internal company documents. "
        "I remember the context of our recent conversation. "
        "Just type your question and I'll do my best to find the answer for you.\n\n"
        "You can use /clear to make me forget our current conversation and start fresh."
    )


async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clears the conversation history for the current chat."""
    chat_id = update.message.chat_id
    if chat_id in chat_histories:
        del chat_histories[chat_id]
        await update.message.reply_text(
            "My memory of our conversation has been cleared."
        )
    else:
        await update.message.reply_text(
            "I have no memory of our conversation to clear."
        )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_question = update.message.text
    chat_id = update.message.chat_id

    history = chat_histories.get(chat_id, [])

    thinking_message = await context.bot.send_message(chat_id, "🧠 Thinking...")

    inputs = {
        "question": user_question,
        "transform_attempts": 0,
        "chat_history": history,
    }

    try:
        final_answer = ""
        final_state = app.invoke(inputs)
        final_answer = final_state.get(
            "generation", "Sorry, I couldn't find an answer."
        )

        new_history = history + [
            HumanMessage(content=user_question),
            AIMessage(content=final_answer),
        ]
        chat_histories[chat_id] = new_history[-MAX_HISTORY_MESSAGES:]

        await context.bot.edit_message_text(
            text=final_answer,
            chat_id=chat_id,
            message_id=thinking_message.message_id,
            parse_mode="HTML",
        )

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        await context.bot.edit_message_text(
            text="An error occurred while processing your request. Please try again.",
            chat_id=chat_id,
            message_id=thinking_message.message_id,
        )


def main() -> None:
    logging.info("Starting bot...")
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("clear", clear_history))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    logging.info("Bot is running. Press Ctrl-C to stop.")
    application.run_polling()


if __name__ == "__main__":
    main()
