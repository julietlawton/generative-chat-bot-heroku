# Create a class for maintaining context to enable multi-turn conversations
class ContextWindow:
    def __init__(self, max_tokens):
        self.token_limit = max_tokens
        self.current_token_count = 0
        self.conversation_turns = []

    # Assemble conversation history
    def get_conversation_history(self):
        # Get the text entries from the list and join into a string
        history = [x[0] for x in self.conversation_turns]
        context = ' '.join(history)
        return context
    
    # Return the current token count
    def get_current_token_count(self):
        return self.current_token_count
    
    # Add an entry to the window
    def add(self, entry, length):
        # Check if adding this entry would put the context window over the token limit
        if self.current_token_count + length > self.token_limit: 
            self.truncate()

        # Add entry to conversation along with its length
        turn = (entry, length)
        self.conversation_turns.append(turn)
        self.current_token_count += length

    # Remove the oldest entry in the conversation
    def truncate(self):
        removed = self.conversation_turns.pop(0)
        removed_token_count = removed[1]
        self.current_token_count -= removed_token_count

        # Make sure we are under the token limit now, if not, run again
        if self.current_token_count > self.token_limit:
            self.truncate()