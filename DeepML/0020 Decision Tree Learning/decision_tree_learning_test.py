from decision_tree_learning import learn_decision_tree

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
    test_decision_tree_learning()
    print("All tests passed")