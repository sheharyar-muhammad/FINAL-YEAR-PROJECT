from django import forms


class UserInputForm(forms.Form):
    user_statement = forms.CharField(
        widget=forms.Textarea(attrs={'placeholder': 'Enter your description'}),

    )

    group = forms.CharField(
        widget=forms.TextInput(attrs={'placeholder': 'Group your statements under a title'}),

    )

    def clean_user_statement(self):
        user_statement = self.cleaned_data['user_statement']
        statements = [s.strip()
                      for s in user_statement.split(';') if s.strip()]

        for statement in statements:
            if len(statement) < 15:  # Checking for 15 characters
                raise forms.ValidationError(
                    "Each statement must be at least 15 characters long.")

        return user_statement
