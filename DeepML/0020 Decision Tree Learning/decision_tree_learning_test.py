import math

from decision_tree_learning import calculate_entropy, calculate_information_gain, learn_decision_tree, majority_class


def test_calculate_entropy_same_labels():
    assert calculate_entropy(['Yes', 'Yes', 'Yes']) == 0.0


def test_calculate_entropy_mixed_labels():
    labels = [
        'No',
        'Yes',
        'Yes',
        'No',
        'Yes',
        'No',
        'Yes',
    ]

    expected = -(3 / 7) * math.log2(3 / 7) - (4 / 7) * math.log2(4 / 7)

    assert math.isclose(calculate_entropy(labels), expected, abs_tol=1e-9)


def test_calculate_information_gain_outlook():
    examples = [
        {'Outlook': 'Sunny', 'Wind': 'Weak', 'PlayTennis': 'No'},
        {'Outlook': 'Overcast', 'Wind': 'Strong', 'PlayTennis': 'Yes'},
        {'Outlook': 'Rain', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
        {'Outlook': 'Sunny', 'Wind': 'Strong', 'PlayTennis': 'No'},
        {'Outlook': 'Overcast', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
        {'Outlook': 'Rain', 'Wind': 'Strong', 'PlayTennis': 'No'},
        {'Outlook': 'Rain', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
    ]

    result = calculate_information_gain(examples, 'Outlook', 'PlayTennis')

    assert math.isclose(result, 0.5916727785823275, abs_tol=1e-9)


def test_majority_class():
    examples = [
        {'PlayTennis': 'No'},
        {'PlayTennis': 'Yes'},
        {'PlayTennis': 'Yes'},
        {'PlayTennis': 'No'},
        {'PlayTennis': 'Yes'},
    ]

    assert majority_class(examples, 'PlayTennis') == 'Yes'


def test_majority_class_tie_breaks_alphabetically():
    examples = [
        {'PlayTennis': 'Yes'},
        {'PlayTennis': 'No'},
    ]

    assert majority_class(examples, 'PlayTennis') == 'No'

def test_decision_tree_learning():
    examples = [
        {'Outlook': 'Sunny', 'Wind': 'Weak', 'PlayTennis': 'No'},
        {'Outlook': 'Overcast', 'Wind': 'Strong', 'PlayTennis': 'Yes'},
        {'Outlook': 'Rain', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
        {'Outlook': 'Sunny', 'Wind': 'Strong', 'PlayTennis': 'No'},
        {'Outlook': 'Overcast', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
        {'Outlook': 'Rain', 'Wind': 'Strong', 'PlayTennis': 'No'},
        {'Outlook': 'Rain', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
    ]
    attributes = ['Outlook', 'Wind']
    target_attr = 'PlayTennis'

    result = learn_decision_tree(examples, attributes, target_attr)

    assert result == {
        'Outlook': {
            'Overcast': 'Yes',
            'Rain': {'Wind': {'Strong': 'No', 'Weak': 'Yes'}},
            'Sunny': 'No',
        }
    }


if __name__ == "__main__":
    test_calculate_entropy_same_labels()
    test_calculate_entropy_mixed_labels()
    test_calculate_information_gain_outlook()
    test_majority_class()
    test_majority_class_tie_breaks_alphabetically()
    test_decision_tree_learning()
    print("All tests passed")