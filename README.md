ΥΦΕ209  | Κβαντική πληροφορία και επεξεργασία

Τα αρχεία είναι:
Πετρίδης πληροφορία παρουσίαση.pdf  
Πετρίδης πληροφορία κώδικας σε py.py  
Πετρίδης πληροφορία κώδικας σε Jupyter.html

Το αρχείο pdf είναι η παρουσίαση του κώδικα που ζητήθηκε για την εξαγωγή των διαγραμμάτων.
Το αρχείο .py είναι ο κώδικας της άσκησης που τρέχει σε .py με το idle της python 3.10.
Το αρχείο .htlm είναι ο κώδικας της άσκησης στο Jupyter. Σημειώνεται πως έγιναν αλλαγές στον κώδικα,
διότι επέστρεφε διαφορετικά αποτελέσματα στο jupyter από ότι τρέχοντας το με το idle της python. Αυτό πιθανώς
ωφείλεται στο γεγονός ότι κάποια δεδομένα να αποθηκεύονται με διαφορετικό τρόπο. 
Μετά τις αλλαγές (odeint -> solve_ivp & quad -> simpson),
οι δύο κώδικες επιστρέφουν το ίδιο.

Στην παρούσα εργασία δημιουργήθηκαν 5 συναρτήσεις, που χρησιμοποιούνται για την λύση της διαφορικής και τις ολοκληρώσεις:
model(x,y), hak_min(x,theta,n,wmin), hak(x,omega,theta,n), Mass(x,theta,n), S(omega,theta,n,wmin).
Για την λύση της Lane Emden χρησιμοποιήθηκε μέθοδος RK4(5) με την solve_ivp και η odeint.
Χρησιμοποιήθηκε επίσης interpolation (interp1d της scipy.interpolate) και extrapolation για τα fill_values.
Για την επίλυση των ολοκληρωμάτων χρησιμοποιήθηκαν οι quad και simpson. Κύρια διαφορά τους εκτός της ακρίβεια είναι η ταχύτητα. 
Τα διαγράμματα έγιναν μέσω του pyplot του matplotlib. 
Τα διαγράμματα που περιέχουν/βρήκαν τα προγράμματα είναι:
.)τα 3 ζητούμενα figure 1, figure 2 και figure 5 του paper,
.)το διάγραμα των λύσεων της Lane Emden για διάφορες τιμές του γ,
.)τα Figure 3 και Figure 4 του paper.
