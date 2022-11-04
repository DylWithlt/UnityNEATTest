using System;
using UnityEngine;
using UnityEngine.UI;

namespace UnitySharpNEAT
{
    public class GuiController : MonoBehaviour
    {
        [SerializeField]
        private NeatSupervisor _neatSupervisor;

        [SerializeField] private Button Start;
        [SerializeField] private Button Stop;
        [SerializeField] private Button RunBest;
        [SerializeField] private Button DeleteSave;
        [SerializeField] private Text InfoText;
        [SerializeField] private GameObject ParamsContentPane;

        [SerializeField] private GameObject ParamListingPrefab;
        void Awake()
        {
            PopulateParams();
        }

        private void PopulateParams()
        {
            createSetting("Gravity", 9.8f, new Vector2(0, 20), onUpdateCallback: (float newValue) =>
            {
                Physics.gravity = new Vector3(0, -newValue, 0);
            });
        }

        private void createSetting(string name, float defaultValue, Vector2 range, Action<float> onUpdateCallback)
        {
            GameObject newList = Instantiate(ParamListingPrefab, ParamListingPrefab.transform.position, ParamListingPrefab.transform.rotation) as GameObject;
            Text label = newList.transform.GetChild(1).GetComponent<Text>();
            Slider slider = newList.transform.GetChild(0).GetComponentInChildren<Slider>();
            InputField input = newList.transform.GetChild(0).GetComponentInChildren<InputField>();

            label.text = name;
            slider.value = defaultValue;
            slider.minValue = range.x;
            slider.maxValue = range.y;
            input.text = defaultValue.ToString();

            slider.onValueChanged.AddListener(delegate
            {
                input.text = slider.value.ToString();
                onUpdateCallback(slider.value);
            });
            input.onEndEdit.AddListener(delegate
            {
                float inputF;
                try {
                    inputF = float.Parse(input.text);
                } catch (Exception)
                {
                    input.text = slider.value.ToString();
                    return;
                };

                slider.value = inputF;
                onUpdateCallback(inputF);
            });

            newList.transform.SetParent(ParamsContentPane.transform, false);
        }

        public void StartNEAT()
        {
            _neatSupervisor.StartEvolution();
        }

        public void StopNEAT()
        {
            _neatSupervisor.StopEvolution();
        }

        public void RunBestAgent()
        {
            _neatSupervisor.RunBest();

        }

        public void DeleteNEATFiles()
        {
            ExperimentIO.DeleteAllSaveFiles(_neatSupervisor.Experiment);
        }

        private void OnGUI()
        {
            InfoText.text = string.Format("Generation: {0}\nFitness: {1:0.00}", _neatSupervisor.CurrentGeneration, _neatSupervisor.CurrentBestFitness);
        }
    }
}

