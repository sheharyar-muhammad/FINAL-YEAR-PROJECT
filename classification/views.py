from django.views.generic import TemplateView, FormView
from django.urls import reverse_lazy
from .forms import UserInputForm
from .ML.final import combine_predictions_weighted
from dashboard.models import RECORD


class UserInputView(FormView):
    template_name = 'dashboard/main.html'
    form_class = UserInputForm
    success_url = reverse_lazy('result')

    def form_valid(self, form):
        user_statement = form.cleaned_data['user_statement']
        grouped = form.cleaned_data['group']

        # Split the input into individual statements based on semicolons (;)
        statements = [s.strip()
                      for s in user_statement.split(';') if s.strip()]

        # Clear previous session data to avoid duplication
        self.request.session['user_statements'] = []
        self.request.session['combined_predictions'] = []

        # Retrieve the lists from the session (which are now empty)
        all_statements = self.request.session['user_statements']
        all_predictions = self.request.session['combined_predictions']

        # Process each statement separately
        for statement in statements:
            combined_prediction = combine_predictions_weighted(statement)

            # Create a Record instance to save user input and prediction
            RECORD.objects.create(
                title='User Input',
                content=statement,
                group=grouped,
                user=self.request.user,
                prediction=combined_prediction
            )

            # Append the new statement and prediction to the respective lists
            all_statements.append(statement)
            all_predictions.append(combined_prediction)

        # Update the session variables with the updated lists
        self.request.session['user_statements'] = all_statements
        self.request.session['combined_predictions'] = all_predictions

        return super().form_valid(form)


class ResultView(TemplateView):
    template_name = 'dashboard/results.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        user_statements = self.request.session.get('user_statements', [])
        predictions = self.request.session.get('combined_predictions', [])

        # Ensure user_statements and predictions are in sync
        if len(user_statements) != len(predictions):
            raise ValueError(
                "Mismatch between the number of statements and predictions.")

        context['user_predictions'] = zip(user_statements, predictions)
        return context
