

const RawData = (props) => {

  const {data} = props;
  if (!data) {
    return (
      <div className="no-data">
        <p><strong>Oops! No data to display.</strong></p>
      </div>
    )
  }

  const [selectedIndex, setSelectedIndex] = React.useState(0);
  const [content, setContent] = React.useState();

  const loadJson = () => {
    fetch(
      data[selectedIndex]["file"],
      {
        headers : { 
          'Content-Type': 'application/json',
          'Accept': 'application/json'
         }
      }
    )
    .then( (response) => {
      if (response.ok) {
        return response.json();
      }
      throw new Error("Content not found!");
    })
    .then(json => setContent(json))
    .catch((err) => {
      setContent(null);
      console.log(err);
    })
  };

  React.useEffect( () => {
      loadJson();
    },
    [selectedIndex]
  );

  return (
    <div className="raws-container">
      <div className="sidebar">
        <ul>
          {data.map( (item, index) => (
            <li
              key={index}
              className={selectedIndex===index ? "selected": null}
              onClick={() => setSelectedIndex(index)}
            >
              {item.label}
            </li>
          ))}

        </ul>
      </div>
      <div className="selected-raw">
        <pre>{JSON.stringify(content, null, 2)}</pre>
      </div>
    </div>
  );
}

const Visualization = (props) => {
  const {images} = props;
  return (
    <div className="images-container">
      {images.map((im, ix) => (
          <img loading="lazy" key={ix} src={im.file} />
      ))}
    </div>
  );
}

const App = (props) => {
  const {title, raws, images} = props;
  const tabs = ["Visualization", "Raw Data"];

  const [currentTab, setCurrentTab] = React.useState(0);

  return (
    <>
      <h2>{title}</h2>
      <div className="tabs">
      {tabs.map( (tab, ix) => (
        <button
          key={ix}
          className={"tab-button" + (currentTab===ix ? " active" : "")}
          onClick={() => setCurrentTab(ix)}
        >
          {tab}
        </button>
      ))}
      </div>
      {currentTab === 0
        ? <Visualization images={images} />
        : <RawData data={raws}/>
      }
    </>
  );
}
